FROM python:3.7
RUN apt-get update && python3 -m pip install --upgrade pip && python3 -m pip install --upgrade Pillow && \
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -U aiogram
COPY telegram_bot/main.py NST.py config.ini ./app/
COPY telegram_bot/gans/ ./app/gans/
WORKDIR ./app/
