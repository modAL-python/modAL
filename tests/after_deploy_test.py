import os

for example_test in os.listdir('example_tests'):
    print('Running ' + example_test + '...')
    os.system('python3 '+ os.path.join('example_tests', example_test))
    print(example_test + ' successful!')