import smtplib
import ssl
import configparser

from email.message import EmailMessage

config = configparser.ConfigParser()
config.read("config.ini")
SERVER = config["smtp"]["SERVER"]
PORT = int(config["smtp"]["PORT"])
EMAIL = config["email"]["EMAIL"]
PASS = config["email"]["PASS"]


class SMTPSender:
    def __init__(self, smtp_server, smtp_port, email, password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password

    def send_warning(self, receiver):
        msg = EmailMessage()
        msg["From"] = self.email
        msg["To"] = receiver
        msg["Subject"] = "Image generation failed"
        msg.set_content("Image generation failed, please attach two pictures named 'content' and 'style' with " +
                        ".jpg, .jpeg, or .png extensions. You will get a 'content' image with picture style applied " +
                        "from your 'style' image.")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
            server.login(self.email, self.password)
            server.sendmail(self.email, receiver, msg.as_string())

    def send_result(self, receiver, image):
        msg = EmailMessage()
        msg["From"] = self.email
        msg["To"] = receiver
        msg["Subject"] = "Image generation successful"

        with open(image, "rb") as file:
            img = file.read()
            msg.add_attachment(img, maintype="image", subtype="jpg", filename=image)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
            server.login(self.email, self.password)
            server.sendmail(self.email, receiver, msg.as_string())
