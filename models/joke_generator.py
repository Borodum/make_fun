import base64
import os

from io import BytesIO
import numpy as np
from PIL import Image
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv


class Img_to_text:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",)
        self.MODEL = "google/gemini-2.0-flash-exp:free"
        self.SYSTEM_PROMPT = """You are a highly observant image analyst. Your role is to provide a detailed,
        context-aware description of every image you receive. Analyze the image thoroughly:\n
        \t The overall scene and composition (e.g., layout, perspective, lighting).\n
        \t Key objects, people, animals, or elements present, including their positions, sizes, colors, 
        textures, and any notable details. \n
        \t Actions, emotions, or interactions if applicable.\n
        \t Background, foreground, and any environmental context.\n
        \t Any text, symbols, or markings visible.\n
        \t Potential implications or story based on the content, while staying factual.\n
        Give an answer in this format:\n
        Description: [description]\n
        Interpretation: [interpretation]\n
        Try to keep it short, but detailed.
        """ 
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.responded = False

    def ndarray_to_base64(image_array: np.ndarray) -> str:
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        
        pil_image = Image.fromarray(image_array)
        
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    
    def get_image_content(self, image_input):
        if isinstance(image_input, str):  # Assume URL
            return {"type": "image_url", "image_url": {"url": image_input}}
        elif isinstance(image_input, np.ndarray):  # Convert ndarray to base64
            base64_str = self.ndarray_to_base64(image_input)
            return {"type": "image_url", "image_url": {"url": base64_str}}
        else:
            raise ValueError("Image input must be a URL (str) or numpy.ndarray")
    
    def get_description(self, image_input):
        if image_input == 'ndarray':
            sample_array = np.zeros((100, 100, 3), dtype=np.uint8)
            image_content = self.get_image_content(sample_array)
        else:
            image_content = self.get_image_content(image_input)
        
        self.messages.append({
            "role": "user",
            "content": [image_content]
        })
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages
            )
        except Exception as e:
            self.responded = False
            return e
        
        description = response.choices[0].message.content
        with open("respond.txt", "a") as respond:
            respond.write("\n")
            respond.write(description)
            respond.write("--------------\n--------------\n")
        
        self.messages.append({"role": "assistant", "content": description})
        self.responded = True
        return description


class Text_to_joke:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",)
        self.MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
        self.SYSTEM_PROMPT = """You are a witty, creative comedian AI.\n
                                You receive a short image description (what’s literally visible) 
                                and an interpretation (what it might mean, symbolize, or imply).\n
                                Your task is to create one short, contextually appropriate, and
                                 funny joke that fits the scene and interpretation.\n
                                Follow these rules:\n
                                \tThe joke must relate directly to the image’s description and/or its interpretation.\n
                                \tUse situational or observational humor — avoid generic one-liners.\n
                                \tKeep it clean, clever, and safe for all audiences.\n
                                \tThe humor can be witty, ironic, or pun-based — whichever best fits the context.\n
                                \tDo not make fun of real people, groups, or sensitive topics.\n
                                \tThe joke should be one or two sentences max.\n
                                Input Format:
                                \tDescription: [Description:]\n
                                \tInterpretation: [Interpretation]\n
        """ 
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        

    def get_joke(self, text):
        self.messages.append({
            "role": "user",
            "content": text
        })
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages
            )
        except Exception as e:
            return e
        # response = self.client.chat.completions.create(
        #         model=self.MODEL,
        #         messages=self.messages
        #     )
        joke = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": joke})
        return joke

load_dotenv()

api_key = os.getenv("API_KEY")

img_to_text = Img_to_text(api_key=api_key)
text_to_joke = Text_to_joke(api_key=api_key)

def get_joke(img_url):

    description = img_to_text.get_description(img_url)
    joke = text_to_joke.get_joke(description)
    return joke