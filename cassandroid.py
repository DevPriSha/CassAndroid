from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import discord
import pandas as pd
from variables import TOKEN_CASS

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
model = AutoModelWithLMHead.from_pretrained('cass-large-four')

chat_history_ids = None

df = pd.read_csv('history.csv')
#create a dictionary with id as key and its frequency as value
id_dict = {}
for i in df['id']:
    if i in id_dict:
        id_dict[i] += 1
    else:
        id_dict[i] = 1


class MyClient(discord.Client):
    def __init__(self):
        super().__init__()

    def query(self, msg, id):
        global id_dict
        global chat_history_ids
        if id not in id_dict:
            id_dict[id] = 0
            step = 0
        elif id_dict[id] == 2:
            id_dict[id] = 0
            step = 0
        else:
            id_dict[id] += 1
            step = id_dict[id]
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(msg + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
         
        chat_history_ids = model.generate(
            bot_input_ids, max_length=200,
            pad_token_id=tokenizer.eos_token_id,  
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature=0.8
        )
    
    # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    async def on_ready(self):
        # print out information when the bot wakes up
        await self.change_presence(activity=discord.Game('with Mara\'s head.'))
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')
        # send a request to the model without caring about the response
        # just so that the model wakes up and starts loading
        self.query('Hello!','0')

    async def on_message(self, message):
        """
        this function is called whenever the bot sees a message in a channel
        """
        # ignore the message if it comes from the bot itself
        if message.author.id == self.user.id:
            return
        if not message.content.startswith('-'):
            return
        # form query payload with the content of the message
        print(message.content)
        #add message.author.id and message.content to a csv file
        df = pd.read_csv('history.csv')
        df = df.concat({'id': message.author.id, 'message': message.content}, ignore_index=True)
        df.to_csv('history.csv', index=False)

        # while the bot is waiting on a response from the model
        # set the its status as typing for user-friendliness
        async with message.channel.typing():
          response = self.query(message.content[1:], message.author.id)
        bot_response = response
        
        # we may get ill-formed response if the model hasn't fully loaded
        # or has timed out
        if not bot_response:
            if 'error' in response:
                bot_response = '`Error: {}`'.format(response['error'])
                print(response)
            else:
                bot_response = 'Hmm... something is not right.'

        # send the model's response to the Discord channel
        print(bot_response)
        await message.reply(bot_response, mention_author=False)

def main():
    client = MyClient()
    client.run(TOKEN_CASS)

if __name__ == '__main__':
  main()
