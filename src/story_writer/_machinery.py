import torch

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
                  image_prompt=image_prompt,
                  text_model=model,
                  image_model=image_model)

    os.remove(temp_file)


def create_story_prompt(outline, num_sentences=7, language='English', target_audience='12-year old kids'):
    """Creates a story prompt from the given outline, specified length, language and target audience."""
    request = f"""
    Write a {num_sentences} sentence story in {language} language for {target_audience}. 
    The story should be a little bit funny and in general written with a positive mood.
    
    This is the outline of the story:
    {outline}
    """

    return request


def create_image_prompt(story, image_type="picture"):
    """Creates an image prompt from the given story."""
    return f"""
Draw a {image_type} about the following story. 
The {image_type} should show realistic looking actors.
The general mood of the {image_type} is positive.
Do not write any text or speech bubbles into the {image_type}.

This is the story:
{story}

Again, keep in mind that the {image_type} should be realistic looking, contain no text, and the general mood should be positive.
"""


def prompt(user_prompt, system_prompt="", model="gpt-4-1106-preview"):
    """A prompt helper function that sends a message to openAI
    and returns only the text response.
    """
    print("Text generation model:", model)
    if model.startswith("gemini"):
        return prompt_vertexai(user_prompt, system_prompt, model)
    elif model.startswith("gpt"):
        return prompt_openai(user_prompt, system_prompt, model)
    else:
        return prompt_ollama(user_prompt, system_prompt, model)


def prompt_ollama(user_prompt, system_prompt, model="llama2"):
    from openai import OpenAI

    # assemble prompt
    system_message = [{"role": "system", "content": system_prompt}]
    user_message = [{"role": "user", "content": user_prompt}]

    # init client
    client = OpenAI(base_url='http://localhost:11434/v1', api_key="none")

    # retrieve answer
    response = client.chat.completions.create(
        messages=system_message + user_message,
        model=model
    )
    reply = response.choices[0].message.content

    return reply


def prompt_vertexai(user_prompt, system_prompt, model="gemini-pro"):

    from vertexai.preview.generative_models import (
        GenerationConfig,
        GenerativeModel,
        Image,
        Part,
        ChatSession,
    )

    my_prompt = f"""
           {system_prompt}

           # Task
           This is the task:
           {user_prompt}
           """

    client = GenerativeModel(model)

    response = client.generate_content(my_prompt).text

    return response


def prompt_openai(user_prompt, system_prompt, model="gpt-4-1106-preview"):
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
    print("Image generation model", model)
    if model.startswith("dall-e"):
        return draw_dall_e_image(prompt, size_str, model)
    elif model.startswith("google"):
        return draw_google_image(prompt, size_str, model)
    else:
        return draw_stable_diffusion_image(prompt, "512x512", model)


def draw_google_image(prompt, size_str, model):

    # modified from: https://codelabs.developers.google.com/generate_creatives_google_ads#5

    import requests
    import json
    import base64
    import os
    from skimage.io import imread
    import io

    image_count = 1

    import os

    # retrieve google-cloud access token
    command = "gcloud auth print-access-token"
    process = os.popen(command)
    access_token = process.read().strip("\n")
    process.close()

    # access_token = os.getenv('GCLOUD_API_KEY')
    project_id = os.getenv('GCLOUD_PPROJECT_ID')

    print("token", access_token)

    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/imagegeneration:predict"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    data = {
        "instances": [
            {
                "prompt": prompt
            }
        ],
        "parameters": {
            "sampleCount": image_count
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()

        for prediction in response_data.get('predictions', []):
            image_data = base64.b64decode(prediction['bytesBase64Encoded'])

    else:
        raise RuntimeError("Request failed:", response.status_code, response.text)

    # Convert to a binary stream
    png_stream = io.BytesIO(image_data)

    # Read the image file into a numpy array
    image_array = imread(png_stream, plugin='imageio')

    return image_array


def draw_stable_diffusion_image(prompt, size_str="512x512", model="stabilityai/stable-diffusion-2-1-base"):

    import numpy as np
    from diffusers import DiffusionPipeline

    size = [int(s) for s in size_str.split("x")]
    width = size[0]
    height = size[1]

    pipe = DiffusionPipeline.from_pretrained(
        model, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    image = pipe(prompt,
                 width=width,
                 height=height
                 ).images[0]

    image_np = np.array(image)

    torch.cuda.empty_cache()

    return image_np


def draw_dall_e_image(prompt, size_str="1024x1024", model='dall-e-3'):
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


