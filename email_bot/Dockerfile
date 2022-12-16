FROM python:3.7
RUN apt-get update && python3 -m pip install --upgrade pip && python3 -m pip install --upgrade Pillow && \
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
COPY email_bot/imap.py email_bot/smtp.py NST.py config.ini ./app/
WORKDIR ./app/
