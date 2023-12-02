import os
import runpy
import sys

import streamlit as st
import story_writer


def main() -> None:
    streamlit_script_path = os.path.join(os.path.dirname(story_writer.__file__), "_streamlit_script.py")
    sys.argv = ["streamlit", "run", streamlit_script_path ]
    runpy.run_module("streamlit", run_name="__main__")

@st.cache_resource
def compute_story(outline, num_sentences, target_audience, language, useless_number):
    from story_writer import prompt, create_story_prompt

    model = "gpt-4-1106-preview"

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

@st.cache_resource
def compute_image(story):
    from story_writer import create_image_prompt, draw_image, package_story

    image_model = "dall-e-3"

    # Generate image
    image_prompt = create_image_prompt(story)
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
        st.title("Story writer")
        st.write("Let AI write you a story.")

        outline = st.text_area("Short story content:", "Cat meets mouse")
        num_sentences = st.number_input("Story length (in sentences):", min_value=3, max_value=100, value=7)
        target_audience = st.text_input("Target audience:", "12-year old kids")
        language = st.selectbox("Language:", ["English", "German", "French", "Klingon"])

        #st.text_input("Language:", "English")

        create_image = st.checkbox("Create image")
        explain = st.checkbox("Explain how it's made")

        ok_button = st.button("Create story")

        cache_seed = st.number_input("Cache seed:", min_value=0, max_value=1000000, value=42)

    # Display the output image in the right column
    with col2:
        if ok_button:
            # Call the function to create the NumPy image
            # numpy_image = convert_to_numpy_image(text1, number, text2)
            pdf_filename = "temp.pdf"

            story, title, story_prompt = compute_story(str(outline), num_sentences, target_audience, language, cache_seed)

            st.title(title)
            st.markdown(story)

            if create_image:
                image, image_prompt = compute_image(story)

                print("hello world")

                left_co, cent_co, last_co = st.columns([1,3,1])
                with cent_co:
                    st.image(image, caption=title, width=600)
            else:
                image = None
                image_prompt = None

            st.write("Disclaimer: This story has been auto-generated using artificial intelligence. Any resemblance to real persons, living or dead, or actual places or events is purely coincidental and unintentional. Read the documentation of the story-writer Python library to learn more: https://github.com/haesleinhuepf/story-writer")

            if explain:
                st.text_area("The story was generated using the following prompt:", story_prompt)
                if create_image:
                    st.text_area("The image was generated using the following prompt:", image_prompt)

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
                          image_prompt=image_prompt)

            abs_path = os.path.abspath(pdf_filename)

            st.download_button(
                "Download Story", Path(abs_path).read_text(), "story.pdf"
            )



if __name__ == "__main__":
    streamlit_app()


