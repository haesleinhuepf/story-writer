
def generate_story(pdf_filename, outline, num_sentences=10, language='English', target_audience='12-year old kids', model='gpt-4-1106-preview', image_model='dall-e-3', explain_how_its_made=False):

    import os
    from skimage.io import imsave
    from ._make_pdf import package_story

    # Generate the story
    story_prompt = create_story_prompt(outline=outline,
                                       num_sentences=num_sentences,
                                       language=language,
                                       target_audience=target_audience)
    story = prompt(story_prompt, model=model)

    # Determine title from story
    title = prompt(f"Formulate a very short title of the following story: {story}")
    title = title.replace("'", "").replace('"', '')

    # Generate image
    image_prompt = create_image_prompt(story)
    image = draw_image(image_prompt, model=image_model)

    temp_file = "temp_story_write_132323.png"

    # Save the image to the temporary file
    imsave(temp_file, image)

    if not explain_how_its_made:
        story_prompt = None
        image_prompt = None

    # Save the PDF
    package_story(pdf_filename=pdf_filename,
                  title=title,
                  story=story,
                  image_filename=temp_file,
                  story_prompt=story_prompt,
                  image_prompt=image_prompt)

    os.remove(temp_file)

def create_story_prompt(outline, num_sentences=7, language='English', target_audience='12-year old kids'):
    """Creates a story prompt from the given outline, specified length, language and target audience."""
    request = f"""
    Write a {num_sentences} sentence story in {language} language for {target_audience}. 
    The story should be a little bit funny and in general written with a positive mood.
    
    The is the rough content of the story:
    {outline}
    """

    return request


def create_image_prompt(story, image_type="comic strip"):
    """Creates an image prompt from the given story."""
    return f"""
Draw a {image_type} about the following story. 
The {image_type} should show realistic looking actors and be consistent from panel to panel.
The general mood of the {image_type} is positive.
Keep it simple. Draw at maximum four panels.
Use minimal text if necessary.

This is the story:
{story}
"""


def prompt(user_prompt, system_prompt="", model="gpt-4-1106-preview"):
    """A prompt helper function that sends a message to openAI
    and returns only the text response.
    """
    from openai import OpenAI

    # assemble prompt
    system_message = [{"role": "system", "content": system_prompt}]
    user_message = [{"role": "user", "content": user_prompt}]

    # init client
    client = OpenAI()

    # retrieve answer
    response = client.chat.completions.create(
        messages=system_message + user_message,
        model=model
    )
    reply = response.choices[0].message.content

    return reply


def draw_image(prompt, size_str="1024x1024", model='dall-e-3'):
    """Draws an image from the given prompt."""
    from openai import OpenAI

    num_images=1
    
    client = OpenAI()
    response = client.images.generate(
      prompt=prompt,
      n=num_images,
      model=model,
      size=size_str
    )
    return images_from_url_responses(response)

def images_from_url_responses(response, input_shape = None):
    """Turns a list of OpenAI's URL responses into numpy images"""
    from skimage.io import imread
    from skimage import transform
    import numpy as np
    images = [imread(item.url) for item in response.data]

    if input_shape is not None:
        # make sure the output images have the same size and type as the input image
        images = [transform.resize(image, input_shape, anti_aliasing=True, preserve_range=True).astype(image.dtype) for image in images]

        if len(input_shape) == 2 and len(images[0].shape) == 3:
            # we sent a grey-scale image and got RGB images back
            images = [image[:,:,0] for image in images]

    if len(images) == 1:
        # If only one image was requested, return a single image
        return images[0]
    else:
        # Otherwise return a list of images as numpy array / image stack
        return np.asarray(images)


