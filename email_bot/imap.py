#!/usr/bin/env python3

import imaplib
import email
import logging
from email.utils import parseaddr
import os
import time
import configparser

from smtp import SMTPSender
from NST import NSTModel

config = configparser.ConfigParser()
config.read("config.ini")
EMAIL = config["email"]["EMAIL"]
PASS = config["email"]["PASS"]
IMAP_SERVER = config["imap"]["SERVER"]
IMAP_PORT = int(config["imap"]["PORT"])
SMTP_SERVER = config["smtp"]["SERVER"]
SMTP_PORT = int(config["smtp"]["PORT"])

IMG_FOLDER = "images"

logging.basicConfig(level=logging.INFO)


def check_image(name: str):
    return (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")) and \
           (name.split(".")[0] == "content" or name.split(".")[0] == "style")


smtp_sender = SMTPSender(SMTP_SERVER, SMTP_PORT, EMAIL, PASS)

imap = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
imap.login(EMAIL, PASS)

if __name__ == '__main__':
    if not os.path.isdir(IMG_FOLDER):
        os.mkdir(IMG_FOLDER)

    while True:
        imap.select("INBOX")
        response_code, messages = imap.search(None, "(Unseen)")

        if response_code == "OK":
            msg_list = messages[0].split()
            if len(msg_list) > 0:
                for i in msg_list:
                    logging.info("Processing new email")
                    res, msg = imap.fetch(i, "(RFC822)")

                    for response in msg:
                        if isinstance(response, tuple):
                            msg = email.message_from_bytes(response[1])
                            sender = parseaddr(msg.get("From"))[1]

                            images = {}
                            if msg.is_multipart():
                                for part in msg.walk():
                                    content_disposition = str(part.get("Content-Disposition"))
                                    if "attachment" in content_disposition:
                                        filename = part.get_filename()
                                        if filename and check_image(filename):
                                            filepath = os.path.join(IMG_FOLDER, filename)
                                            images[filename.split(".")[0]] = filepath
                                            open(filepath, "wb").write(part.get_payload(decode=True))

                            if "content" not in images or "style" not in images:
                                smtp_sender.send_warning(sender)
                                logging.info(f"Sender: {sender}, generation failed")
                            else:
                                model = NSTModel(content_img=images["content"], style_img=images["style"])
                                model.fit()
                                output = model.get_image()
                                output.save("output.jpg")
                                smtp_sender.send_result(sender, "output.jpg")
                                logging.info(f"Sender: {sender}, generation successful")
            else:
                logging.info("No new emails to process, sleeping for a minute")
                time.sleep(60)
