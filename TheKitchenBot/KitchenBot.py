###############################################################################
#                             The Maya vitual assistant                                         #
###############################################################################
# Setup
## for speech-to-text
import speech_recognition as sr

## for text-to-speech
from gtts import gTTS

## for language model
import transformers

## for data
import os
import datetime
import numpy as np

# open the recipe files
ing_file = open(os.path.join('Recipes', 'french pound cake', 'ingredients.txt'), "r")
inst_file = open(os.path.join('Recipes', 'french pound cake', 'instructions.txt'), "r")

# create lists
ing_list = ing_file.readlines()
inst_list = inst_file.readlines()

# close files
ing_file.close()
inst_file.close()


# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            print("me -->  ERROR")

    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        os.system("afplay res.mp3")  #mac->afplay | windows->start
        os.remove("res.mp3")

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


# Run the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="maya")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    while True:
        ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = '''Hi. My name is Maya. I am your new culinary assistant,
           I can help you get around in the kitchen. 
           Right now, I am not very socially competent, but hopefully, you can make me more natural to interact with.'''
        

        ### Here is where you can hardcode responses 

        ## ask for the time
        elif "time" in ai.text:
            res = ai.action_time()
        
        ## Baking instructions 
        elif "bake" in ai.text:
            res = '''So far, I only know one recipe: French Pound Cake. 
           You can ask me for the list of Ingredients by saying Ingredients. 
           You can ask me for the list of instructions by saying Instructions'''


        # the list of ingredients is a .txt file in the Recipe folder and is loaded in line 20 
        elif "ingredients" in ai.text:
            active_list = ing_list
            i = 0
            res = "I will now read ingredients list, say next to start"
        
        # the list of instructions is also a .txt file in the Recipe folder and is loaded in line 21 
        elif "instructions" in ai.text: 
            active_list = inst_list
            i = 0
            res = "I will now read the instructions list, say next to start"    

        # the command "next" will proceed to the next line of the ingredient/instruction text file loaded as the "active_list" 
        elif "next" in ai.text:
            res = active_list[i]
            i += 1

        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!"])
        
        ### Free conversation engine 
        
        else:   
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_speech(res)