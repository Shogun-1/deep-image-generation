# Image Generation Bots
This project implements image generation by bots.
There are 2 types of bots supported: Telegram bot and email bot. 
You can use any of them, please look at instructions on how to use each bot below.  

## Overview
### Features and supported tasks
You can use bots for the following range of tasks:
1. **Neural Style Transfer**.   
You need to send two pictures: the first one titled as `content` picture, 
while the second image titled as `style` picture. After generation, you will get your `content` picture
with the style applied from your `style` picture.  
2. **Neural Style Transfer with style blending**.  
The same as the basic NST, but you need to send two `style`
pictures instead of one. Of course, `content` image is required as well. 
Style from the first `style` picture will be applied to the upper-right part of the `content` image, 
while style of the second `style` picture will be applied to the lower-left part of the `content` image. 
3. **Image Generation via Generative Adversarial Networks (GANs)**.  
Here you can upload an image and get a modified version of your input image. 
For example, you can use this module to get zebra picture from input horse picture.
For more guidance on how to perform such operations, please follow the instructions from the bot.  

### Bots
**Telegram Bot**  
* It is recommended to use this bot, because it offers much wider functionality rather than email bot.
The bot supports all the tasks described above. 
* You send and receive images through Telegram: **@some_power_bot**.  
* Instructions are below.

**Email bot**  
* The bot supports only **Neural Style Transfer** module. 
* You communicate with bot via email: **bot.power@mail.ru**.  
* Instructions are below.

## Instructions on how to use each bot
### Starting servers
Before starting communication with bots, you need to launch the servers. Use `docker` for this.
1. Just type the following command: `docker-compose up --build`.
2. Wait for both servers (Telegram bot and email bot) to boot up.
Check logs for information when both servers are ready. 
The first start-up will be much longer than subsequent ones. In most cases, it takes about 5-7 minutes.
3. You are good to go! Servers are running, and now you can interact with bots. Instructions are given below.  

### Telegram Bot
* Telegram Bot supports all the tasks described above in the `Overview` section.  
* Search for PowerBot(**@some_power_bot**) in Telegram. 
* Type `/hello` or `/start` to begin interactive session.
* Follow the instructions from the bot. These are highly clear and detailed, the bot will guide you on every step.

### Email bot
* Email bot supports only **Neural Style Transfer** module.
* Attach two images to the letter and send it to the following email address: **bot.power@mail.ru**.
    - The attached images must be named `content` and `style`.
    - You can send images only with the `.jpg`, `.jpeg`, or `.png` extensions.
    - For example, you can send `content.png` and `style.jpg` images attached.
* Subject and body can be empty or random. It doesn't matter.
* Wait for processing your request. In most cases, it takes no more than 5 minutes.
* After processing, you will receive a letter to the email you used to send images. 
It will contain your generated image. In case you haven't sent 2 images as required, 
you will get a notification letter on what you should send to the bot.

## How to configure and start your own bot
There are some guidelines that can be helpful if you want to start your own bot.
### Email bot
- For email bot, you need to create a new inbox address using any domain you like (i.e., gmail).  
- Then you need to find out host addresses and ports for SMTP (sending emails) and IMAP (reading emails)
protocols for your email provider. You can use this [link](https://www.systoolsgroup.com/imap/) to find them out.
- After you found it, just update [config file](/config.ini):
    - Use your full email address and password (usually you need to use a special App Password for email as the main password will not work,
please look [here (for Gmail)](https://support.google.com/accounts/answer/185833?hl=en#zippy=)).
    - Don't forget to update your IMAP and SMTP hosts and ports in the config.
- You're ready to go!

### Telegram bot
- Here, you need to create your own bot for Telegram.
- For example, you can use [this guide](https://sendpulse.com/knowledge-base/chatbot/telegram/create-telegram-chatbot) (check only first step, it would be enough).
- After you created the bot, you need to get its token and copy the value to [config file](/config.ini).
- Everything is set up now!


## Closing notes
Thanks for using the bots!   
You can use images from [here](images) for exploration and testing purposes.  
