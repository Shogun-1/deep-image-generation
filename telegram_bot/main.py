#!/usr/bin/env python3

import logging
from io import BytesIO

from aiogram import Bot, Dispatcher, executor, types

import NST
from gans.cyclegan import CycleGAN, CYCLEGAN_TASK_LIST
from gans.pix2pix import Pix2Pix, PIX2PIX_TASK_LIST

API_TOKEN = '1505628465:AAF3lLnq58KjaACimrMCJWAoEDs57DpAnpQ'
VERSION = 1.0

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

supported_commands = "The following commands are supported: " + \
                     "\n/nst_info - There is all the basic info you need to know about style transfer operations." + \
                     "\n/gan_info - There is all the info about generating images." + \
                     "\n/nst_start - This will start style transfer operation." + \
                     "\n/nst_2styles_start - This will start 2 styles transfer operation." + \
                     "\n/clear_storage - This will delete all the pictures you sent." + \
                     "\n/creator - This will show the info about my creator." + \
                     "\n/info - This will show the info about me." + \
                     "\n/corgi - This will send a cute corgi photo for you." + \
                     "\n\nType /help to get the list of supported commands again. Have fun!"

img_storage = {}

cyclegan = None
pix2pix = None


def img_to_bio(img):
    bio = BytesIO()
    bio.name = 'output.jpeg'
    img.save(bio, 'JPEG')
    bio.seek(0)
    return bio


def format_task_list(list_):
    out = ' '.join(map(lambda elem: str('/' + elem), list_))
    return out


@dp.message_handler(commands=['start', 'hello'])
async def start_handler(message: types.Message):
    """
    This handler will be called when user sends `/start` or '/hello' command
    """
    name = message.from_user.first_name
    await message.answer(f"Hello, {name}!\n\nI'm PowerBot. Ready to work. " + supported_commands)


@dp.message_handler(commands='help')
async def help_handler(message: types.Message):
    await message.answer(supported_commands)


@dp.message_handler(commands='creator')
async def creator_handler(message: types.Message):
    await message.answer("I was created by Aleksandr Akhmatdianov and I'm happy to be alive :) Feel free to contact "
                         "him via Telegram for any assistance: @aleksandr_akhmatdianov")


@dp.message_handler(commands='info')
async def info_handler(message: types.Message):
    await message.answer("PowerBot version: " + str(VERSION) + "\nBuilt on Python 3, aiogram and Telegram Bot API.")


@dp.message_handler(commands='corgi')
async def corgi_handler(message: types.Message):
    """
    Sending corgi photo to user
    """
    img_out = 'https://drive.google.com/uc?export=download&id=1LqUPrXeUEfPkxf-YArt7gxQetXaNakA1'
    await message.answer_photo(img_out)


@dp.message_handler(commands='nst_info')
async def nst_info_handler(message: types.Message):
    await message.answer('This is Neural Style Transfer module. It supports 2 submodules: basic style transfer and '
                         'style transfer with style blending. \n\nIf you want to use basic style transfer I need to '
                         'get 2 pictures from you. The first one is the picture you want to modify and the second one '
                         'is the picture which style you would like to apply to the first picture. Please send me one '
                         'picture per one message. When it\'s done just type /nst_start command to start the '
                         'operation. \n\nIf you want to use style blending, you need to send me 3 pictures instead of '
                         '2. The first picture is the picture you want to modify. Style of the second picture will be '
                         'applied to the upper-right part of the first picture, while style of the third picture will '
                         'be applied to the lower-left part of the first picture. After you send me 3 images just '
                         'type /nst_2styles_start command to begin image generation. \n\nAfter using either of this '
                         'operations all pictures you sent will be deleted. \n\nWARNING: please don\'t try to upload '
                         'multiple pictures per one message, you need to send one image per message. Moreover, '
                         'you will get detailed instructions about what to do next after you upload each '
                         'consecutive image. Also you can send only 3 images per one session (operation). You can use '
                         '/clear_storage command to delete all the pictures you sent. After that you can send me new '
                         'ones! Process operations take decent amount of time (5+ minutes), so please keep it in mind.')


@dp.message_handler(commands='gan_info')
async def gan_info_handler(message: types.Message):
    msg = str('This is Image Generation module. You can upload an image and get modified version of the '
              'uploaded image. Just upload the image and pick the command you want. Currently available '
              'transformations are ' +
              format_task_list(CYCLEGAN_TASK_LIST) +
              '\nFor example, you can upload picture of horse, then you can use /horse2zebra command and you will get '
              'your picture of zebra. Feel free to experiment!')
    await message.answer(msg)


@dp.message_handler(content_types=types.message.ContentType.PHOTO)
async def image_handler(message: types.Message):
    user_id = message.from_user.id

    if user_id in img_storage and len(img_storage[user_id]) > 3:
        await message.answer('You have already uploaded 3 images. If you want to delete them, type /clear_storage.')
        return

    img_id = message.photo[-1].file_id
    file = await bot.get_file(img_id)
    img_file = await bot.download_file(file.file_path)

    if user_id not in img_storage:
        img_storage[user_id] = {}
        img_storage[user_id]['content'] = img_file
        await message.answer('Thank you! Now you can upload your style image! Or, if you want, you can generate new ' +
                             'picture from uploaded picture via Image Generation module (type /gan_info for details).')
    elif len(img_storage[user_id]) == 1:
        img_storage[user_id]['style1'] = img_file
        await message.answer('Now you can upload the second style image if you want to use style blending or '
                             'you can start style transfer right now via /nst_start command!')
    elif len(img_storage[user_id]) == 2:
        img_storage[user_id]['style2'] = img_file
        await message.answer('Now you can start 2 styles transfer operation via /nst_2styles_start command! If you '
                             'want to start basic style transfer, just type /nst_start command. In this case last '
                             'picture you sent will not be used.')


@dp.message_handler(commands='nst_start')
async def nst_handler(message: types.Message):
    user_id = message.from_user.id
    if user_id not in img_storage or len(img_storage[user_id]) < 2:
        await message.answer('Please upload 2 pictures at first.')
        return

    await message.answer('Style transfer operation started. This operation usually takes around 5 minutes, '
                         'but in rare occasions it can take up to 10 minutes. Please wait.')

    model = NST.NSTModel(content_img=img_storage[user_id]['content'], style_img=img_storage[user_id]['style1'])
    model.fit()
    output = model.get_image()

    bio = img_to_bio(output)

    await bot.send_photo(message.chat.id, photo=bio)
    await message.answer('That\'s it!')

    del img_storage[user_id]
    del model


@dp.message_handler(commands='nst_2styles_start')
async def nst_2styles_handler(message: types.Message):
    user_id = message.from_user.id
    if user_id not in img_storage or len(img_storage[user_id]) < 3:
        await message.answer('Please upload 3 pictures at first.')
        return

    await message.answer('Style transfer with style blending operation started. This operation usually '
                         'takes around 10 minutes, but in rare occasions it can take up to 20 minutes. Please wait.')

    model = NST.NSTModel(content_img=img_storage[user_id]['content'], style_img=img_storage[user_id]['style1'],
                         style_location='upper_right')
    model.fit()
    mid_output = model.get_image()

    await message.answer('Still working... approx. 5 minutes remaining.')

    model = NST.NSTModel(content_img=mid_output, style_img=img_storage[user_id]['style2'],
                         style_location='lower_left')
    model.fit()
    output = model.get_image()

    bio = img_to_bio(output)

    await bot.send_photo(message.chat.id, photo=bio)
    await message.answer('That\'s it!')

    del img_storage[user_id]
    del model


@dp.message_handler(commands=CYCLEGAN_TASK_LIST)
async def cyclegan_handler(message: types.Message):
    user_id = message.from_user.id
    task = message.text[1:]

    if user_id not in img_storage:
        await message.answer('Please upload picture at first.')
        return

    await message.answer('Image generation started. This operation usually takes around 15 seconds, but in rare '
                         'occasions it can take up to 3 minutes. Please wait.')

    global cyclegan
    if cyclegan is None:
        cyclegan = CycleGAN()

    cyclegan.set_task(task)
    output = cyclegan.generate(img_storage[user_id]['content'])

    bio = img_to_bio(output)

    await bot.send_photo(message.chat.id, photo=bio)
    await message.answer('Generated image is ready!')

    del img_storage[user_id]


@dp.message_handler(commands=PIX2PIX_TASK_LIST)
async def pix2pix_handler(message: types.Message):
    user_id = message.from_user.id

    if user_id not in img_storage:
        await message.answer('Please upload picture at first.')
        return

    await message.answer('Image generation started. This operation usually takes around 15 seconds, but in rare '
                         'occasions it can take up to 3 minutes. Please wait.')

    global pix2pix
    if pix2pix is None:
        pix2pix = Pix2Pix()

    output = pix2pix.generate(img_storage[user_id]['content'])

    bio = img_to_bio(output)

    await bot.send_photo(message.chat.id, photo=bio)
    await message.answer('Generated image is ready!')

    del img_storage[user_id]


@dp.message_handler(commands='clear_storage')
async def clear_handler(message: types.Message):
    user_id = message.from_user.id
    img_storage.pop(user_id, None)
    await message.answer('All previously sent pictures were successfully deleted.')


@dp.message_handler()
async def unknown_command_handler(message: types.Message):
    await message.answer("This command is not supported. Type /help for list of available commands.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
