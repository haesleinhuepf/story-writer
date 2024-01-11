import os
import runpy
import sys

import streamlit as st
import story_writer
from story_writer._locale import translate


def main() -> None:
    streamlit_script_path = os.path.join(os.path.dirname(story_writer.__file__), "_streamlit_script.py")
    sys.argv = ["streamlit", "run", streamlit_script_path ]
    runpy.run_module("streamlit", run_name="__main__")

@st.cache_resource
def compute_story(outline, num_sentences, target_audience, language, model= "gpt-4-1106-preview", useless_number=0):
    from story_writer import prompt, create_story_prompt

    # Generate the story
    story_prompt = create_story_prompt(outline=outline,
                                       num_sentences=num_sentences,
                                       language=language,
                                       target_audience=target_audience)

    story = prompt(story_prompt, model=model)

    # Determine title from story
    title = prompt(f"Formulate a very short title of the following story: {story}")
    title = title.replace("'", "").replace('"', '')

    return story, title, story_prompt

def random_outline(language='English', model="gpt-4-1106-preview"):
    from story_writer import prompt

    outline_prompt = f"""
    Write a potentially funny one- or two-sentence story, where things and persons are replaced by place holders.
    Write the story in the {language} language and also the place-holders.
    Do not add any other explanatory text. Respond with just the story please.
    Example 1: A <PERSON> walks into a <PLACE> and asks for a <THING>. Suddenly ...
    Example 2: On a <WEATHER> day, <PERSON> decides for a walk. On its way they find a <THING>.
    """

    return prompt(outline_prompt, model=model)


@st.cache_resource
def compute_image(story, image_model="dall-e-3", image_type="picture"):
    from story_writer import create_image_prompt, draw_image, package_story

    # Generate image
    image_prompt = create_image_prompt(story, image_type=image_type)
    image = draw_image(image_prompt, model=image_model)

    return image, image_prompt

def streamlit_app():
    import numpy as np
    from PIL import Image
    import io

    # hide deploy button
    st.set_page_config(page_title="Story Writer", layout="wide")
    st.markdown("""
        <style>        
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)

    # Define two columns for layout
    col1, col2 = st.columns(spec=[1,2])

    # User input fields in the left column
    with (col1):
        st.title(translate("AI Story writer"))
        st.write(translate("Let AI write you a story."))

        def randomize_outline():
            print(language)
            st.session_state.outline = random_outline(language=language, model=text_model)

        outline = st.text_area(translate("Short story content:"), key='outline', height=200)

        st.button(translate("Randomize"), on_click=randomize_outline)


        num_sentences = st.number_input(translate("Story length (in sentences):"), min_value=3, max_value=100, value=7)
        target_audience = st.text_input(translate("Target audience:"), translate("young adults"))
        language = st.selectbox(translate("Language:"), ["German", "English", "French", "Klingon"])

        create_image = st.checkbox(translate("Generate image"), value=True)
        explain = st.checkbox(translate("Explain how it's made"), value=True)

        ok_button = st.button(translate("Generate story"))

        st.markdown("## " + translate("Advanced options"))

        text_model = st.selectbox(translate("Text generation model:"),
                                   ["gpt-4-1106-preview", "gemini-pro"])

        image_model = st.selectbox(translate("Image generation model:"),
                                   ["dall-e-3", "stabilityai/stable-diffusion-2-1-base", "runwayml/stable-diffusion-v1-5"])
                                       #, "google/imagen"

        image_type = st.selectbox(translate("Image type:"),
                                   ["picture", "photo", "comic", "comic-strip", "scribble"])

        cache_seed = st.number_input("Random number:", min_value=0, max_value=1000000, value=42)

    # Display the output image in the right column
    with col2:
        if ok_button:
            # Call the function to create the NumPy image
            # numpy_image = convert_to_numpy_image(text1, number, text2)
            pdf_filename = "temp.pdf"

            story, title, story_prompt = compute_story(str(outline), num_sentences, target_audience, language, str(text_model), cache_seed)

            st.title(title)
            st.markdown(story)

            if create_image:
                image, image_prompt = compute_image(story, image_model=image_model, image_type=image_type)

                left_co, cent_co, last_co = st.columns([1,3,1])
                with cent_co:
                    st.image(image, caption=title, width=600)
            else:
                image = None
                image_prompt = None

            st.write("Disclaimer: This story has been auto-generated using artificial intelligence. Any resemblance to real persons, living or dead, or actual places or events is purely coincidental and unintentional. Read the documentation of the story-writer Python library to learn more: https://github.com/haesleinhuepf/story-writer")


            if explain:
                st.text_area(f"The story was generated using the following prompt sent to {text_model}:", story_prompt)
                if create_image:
                    st.text_area(f"The image was generated using the following prompt sent to {image_model}:", image_prompt)

            import os
            from pathlib import Path
            from skimage.io import imsave
            from story_writer import package_story

            if not explain:
                story_prompt = None
                image_prompt = None

            if create_image:
                temp_file = "temp_story_write_132323.png"
                imsave(temp_file, image)
            else:
                temp_file = None

            package_story(pdf_filename=pdf_filename,
                          title=title,
                          story=story,
                          image_filename=temp_file,
                          story_prompt=story_prompt,
                          image_prompt=image_prompt,
                          text_model=text_model,
                          image_model=image_model)

            abs_path = os.path.abspath(pdf_filename)

            st.download_button(
                "Download Story", Path(abs_path).read_text(), "story.pdf"
            )



if __name__ == "__main__":
    streamlit_app()


