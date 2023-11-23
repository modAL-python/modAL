"""
This example demonstrates how to use the active learning interface with Pytorch.
The example uses Skorch, a scikit learn wrapper of Pytorch.
For more info, see https://skorch.readthedocs.io/en/stable/

pip install transformers torchvision tqdm
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from modAL.models import ActiveLearner
from PIL import Image
from skorch import NeuralNetClassifier
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoProcessor, CLIPModel

# Setting constants for the dataset sizes
TRAIN_SIZE = 4000
VAL_SIZE = 1000
N_INITIAL = 1000
N_QUERIES = 10
N_INSTANCES = 100


# Determining the device (GPU/CPU) for computation
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Loading a pre-trained ResNet18 model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Setting up the classifier with the NeuralNetClassifier wrapper from skorch
classifier = NeuralNetClassifier(
    model,
    # max_epochs=100,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=None,
    verbose=1,
    device=device,
)

# Defining transformations for the CIFAR10 dataset
transform = transforms.Compose(
    [
        # transforms.Resize(224),  # ResNet18 was originally trained on 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizing the data
    ]
)

# Loading CIFAR10 dataset for training and testing
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset), shuffle=False, num_workers=0
)

# Splitting the dataset into training, validation, and test sets
X, y = next(iter(trainloader))
X = X[: TRAIN_SIZE + VAL_SIZE]
y = y[: TRAIN_SIZE + VAL_SIZE]

X_train, X_val = (
    X[:TRAIN_SIZE],
    X[TRAIN_SIZE:],
)

y_train, y_val = (
    y[:TRAIN_SIZE],
    y[TRAIN_SIZE:],
)
X_test, y_test = next(iter(testloader))

# Loading the CLIP model for feature extraction
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# Extracting features for each image in the dataset using CLIP
features = []
for i, img in tqdm.tqdm(enumerate(X)):
    img = (
        img.permute(1, 2, 0).detach().cpu().numpy()
    )  # Converting image format for processing
    img = (img * 255).astype(np.uint8)  # Rescaling the image pixel values
    image = Image.fromarray(img.squeeze())

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = model.get_image_features(**inputs)
        assert image_features.ndim == 2
        features.append(image_features)

# Concatenating and normalizing the extracted features
embeddings = torch.cat(features)
embeddings /= embeddings.norm(dim=-1, keepdim=True)

# Splitting embeddings for training and validation
train_embeddings = embeddings[:TRAIN_SIZE]
val_embeddings = embeddings[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]

# Initial random selection of samples for active learning
initial_idx = np.random.choice(range(len(X_train)), size=N_INITIAL, replace=False)

X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# Creating the pool of samples for active learning
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

# Removing the initial samples' embeddings from the training set
mask = torch.ones(X_train.size(0), dtype=bool)
mask[initial_idx] = False
train_embeddings = train_embeddings[mask]

# Initializing the Active Learner with the initial dataset
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial,
    y_training=y_initial,
)

# Active learning loop
for idx in range(N_QUERIES):
    # Predicting probabilities for the validation set
    proba_distribution = torch.from_numpy(learner.predict_proba(X_val))
    indices = y_val.reshape(-1, 1)
    proba = torch.gather(
        proba_distribution, 1, indices
    ).squeeze()  # Extracting relevant probabilities
    hardness = 1 - proba  # Calculating the hardness of the samples

    # Calculating similarity between training and validation embeddings
    similarity = (100.0 * train_embeddings @ val_embeddings.T).softmax(dim=-1)
    indirect_hardness = hardness[similarity.argmax(dim=-1).cpu()]
    indirect_hardness /= indirect_hardness.sum()

    # Selecting instances based on calculated hardness
    query_idx = torch.multinomial(indirect_hardness, N_INSTANCES)
    query_instance = X_pool[query_idx]

    # Teaching the learner with the newly selected instances
    learner.teach(X_pool[query_idx], y_pool[query_idx], only_new=True)

    # Removing the selected instances from the pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    mask = torch.ones(train_embeddings.size(0), dtype=bool)
    mask[query_idx] = False
    train_embeddings = train_embeddings[mask]

# Evaluating the final accuracy of the learner
print(learner.score(X_test, y_test))
