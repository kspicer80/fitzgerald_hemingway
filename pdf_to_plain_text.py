import re
import PyPDF2

# Open the PDF file
pdf_file = open('data/hemingway/bell_tolls.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# Get the number of pages in the PDF
pages = pdf_reader.numPages
print(pages)

# Initialize a string to store the text
text = ""

# Iterate through each page and extract the text
for page in range(9, pages):
    page_obj = pdf_reader.getPage(page)
    text += page_obj.extractText()

# Close the PDF file
pdf_file.close()

with open('bell_tolls.txt', 'w') as outfile:
    outfile.write(text)
