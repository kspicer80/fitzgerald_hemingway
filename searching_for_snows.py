'''import os

directory = r'data\hemingway'

search_string = "Snows of Kilimanjaro"

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), encoding='utf-8') as file:
            file_contents = file.read()
            if search_string in file_contents or search_string.lower() in file_contents:
                print(f'The string "{search_string}" was found in {filename}.')
            else:
                print(f'The string "{search_string}" was not found in any of the files.')
                '''

