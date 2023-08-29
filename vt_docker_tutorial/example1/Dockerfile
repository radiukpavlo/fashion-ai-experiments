FROM python:3.8
ADD main.py .
RUN pip install requests beautifulsoup4 python-dotenv
CMD [ "python", "main.py" ]

# FROM node:20-alpine
# RUN apk add --no-cache python3 g++ make
# WORKDIR /app
# COPY . .
# RUN yarn install --production
# CMD ["python", "/app/src/index.js"]
