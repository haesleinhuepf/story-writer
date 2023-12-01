def package_story(pdf_filename, title, story, image_filename, story_prompt, image_prompt):
    """Packages the story into a PDF file."""
    pdf_content = f"""
# {title}

{story}

![]({image_filename})
"""

    if story_prompt is not None and image_prompt is not None:
        pdf_content = pdf_content + f"""
PAGE_BREAK

## How it works

We used your short story notes as a prompt:

```
{story_prompt}
```

Then, we used this story to ask chatGPT for an image using this prompt:

```
{image_prompt}
```

"""
    pdf_content = pdf_content + f"""
Disclaimer: This story has been auto-generated using artificial intelligence. Any resemblance to real persons, living or dead, or actual places or events is purely coincidental and unintentional. Read the documentation of the story-writer Python library to learn more: https://github.com/haesleinhuepf/story-writer
"""

    make_pdf(pdf_filename, pdf_content)


def make_pdf(pdf_filename, text):
    """Makes a PDF file from the given text. The text can contain simple markdown."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    import os
    
    # Check if file exists
    if os.path.exists(pdf_filename):
        os.remove(pdf_filename)

    margin = 1 * inch
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin)
    
    # Styles for paragraphs
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleH1 = styles["Heading1"]
    styleH2 = styles["Heading2"]
    styleH3 = styles["Heading3"]
    styleC = styles["Normal"]
    
    # Split text into two columns
    text_columns = text.split('\n\n')

    # Elements to add to the document
    elements = []
    
    # Add text in two columns
    for col_text in text_columns:
        col_text = col_text.strip()
        if col_text == "PAGE_BREAK":
            elements.append(PageBreak())
        elif col_text.startswith("![]("):
            image_path = col_text[4:-1]
            img = Image(image_path, width=4*inch, height=4*inch)
            elements.append(img)
        elif col_text.startswith("###"):
            elements.append(Paragraph(col_text[3:], styleH3))
        elif col_text.startswith("##"):
            elements.append(Paragraph(col_text[2:], styleH2))
        elif col_text.startswith("#"):
            elements.append(Paragraph(col_text[1:], styleH1))
        elif col_text.startswith("```"):
            elements.append(Paragraph(col_text[3:-3], styleC))
        else:
            elements.append(Paragraph(col_text, styleN))
            
        elements.append(Spacer(1, 12))    
    
    # Build the PDF
    doc.build(elements)

    