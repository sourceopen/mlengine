FROM openjdk:8u181
ADD build/distributions/mlengine-boot*.zip app.zip
RUN unzip app.zip -d /tmp/app && mv /tmp/app/mlengine* /app
WORKDIR /app
CMD ["bin/mlengine", "--spring.profiles.active=RAM" ,"--spring.config.additional-location=file:/data/config/server.properties, file:/data/config/server.yml"]