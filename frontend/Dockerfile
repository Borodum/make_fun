FROM node:18

WORKDIR /app


COPY package*.json ./

RUN npm install

COPY src ./src
COPY index.html .
COPY vite.config.* .
COPY tsconfig*.json .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]