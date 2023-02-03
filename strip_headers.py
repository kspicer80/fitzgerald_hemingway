import os

directory = 'data'
'''
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "rb") as file:
                    data = file.read()
        # process data
            except UnicodeDecodeError as e:
                print(f"Error processing file {filename}: {e}")
'''
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(root, filename)

            with open(file_path, "r", encoding='ISO-8859-1') as file:
                lines = file.readlines()
            start = "*** START OF THE PROJECT GUTENBERG"
            end = "*** END OF THE PROJECT GUTENBERG"
            start_indices = [i for i, line in enumerate(lines) if start in line]
            if not start_indices:
                continue
            start_index = start_indices[0] + 1
            try:
                end_index = lines.index(end)
            except ValueError:
                continue
            stripped_lines = lines[start_index:end_index]
            stripped_file_path = os.path.join(root, 'stripped_' + filename)
            with open(stripped_file_path, 'w') as stripped_file:
                stripped_file.writelines(stripped_lines)




