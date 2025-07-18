from PyPDF2 import PdfReader, PdfWriter

INPUT_PATH = "paper_results/cpp/case_study_1_figure_1.pdf"
OUTPUT_PATH = INPUT_PATH.replace(".pdf", "_cropped.pdf")

# Load the PDF
reader = PdfReader(INPUT_PATH)
writer = PdfWriter()

# Get the first page
page = reader.pages[0]

# Define crop coordinates (adjust these values as needed)
original_width = page.mediabox.width
original_height = page.mediabox.height

# Crop to include the top 25% of the page (assuming 4 rows of subplots)
# Modify the divisor (4) if the layout differs
crop_height = original_height / 1.95
page.mediabox.lower_left = (0, original_height - crop_height)
page.mediabox.upper_right = (original_width, original_height)

# # crop width
# crop_width = original_width / 2
# page.mediabox.lower_left = (original_width / 2, original_height - crop_height)
# page.mediabox.upper_right = (original_width, original_height)

# Add the cropped page to the writer
writer.add_page(page)

# Save the output
with open(OUTPUT_PATH, "wb") as f:
    writer.write(f)