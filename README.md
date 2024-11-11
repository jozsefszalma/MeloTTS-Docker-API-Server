# MeloTTS Docker API Server

A quick easy way to access [MeloTTS](https://github.com/myshell-ai/MeloTTS) through REST API calls.

## Build Image
Assuming you have docker installed and setup.

(This might take a bit because MeloTTS is a big dependency)
#### Local

    git clone git@github.com:timhagel/MeloTTS-Docker-API-Server.git
    cd MeloTTS-Docker-API-Server
    docker build -t timhagel/melotts-api-server .

#### Docker Hub

    docker pull timhagel/melotts-api-server
    
## Languages and Speakers

#### Language

- EN - English
- ES - Spanish
- FR - French
- ZH - Chinese
- JP - Japanese
- KR - Korean

#### Speaker IDs

- EN-US - American English accent
- EN-BR - British English accent
- EN_INDIA - Indian English accent
- EN-AU - Australian English accent
- EN-Default - Default English accent
- **Notice!** Currently only English accents are working, and other accents are returning an error. This does not mean that other languages do not work!

## Running

### Run (CPU) (English)

    docker run --name melotts-server -p 8888:8080 -e DEFAULT_SPEED=1 -e DEFAULT_LANGUAGE=EN -e DEFAULT_SPEAKER_ID=EN-Default timhagel/melotts-api-server

### Run (GPU) (English)
    
    docker run --name melotts-server -p 8888:8080 --gpus=all -e DEFAULT_SPEED=1 -e DEFAULT_LANGUAGE=EN -e DEFAULT_SPEAKER_ID=EN-Default timhagel/melotts-api-server

### Model Timeout (Optional)

The memory consumed by the model can be freed up periodically by adding MODEL_IDLE_TIMEOUT environment variable and setting it in seconds. 
This environment variable defaults to -1; this keeps the model in memory indefinitely.
    
    -e MODEL_IDLE_TIMEOUT=600 (keeps model in memory for 10 minutes)

## Call API

**localhost:8888/convert/tts**

### Use Environment Defaults
Response: .wav

###### Post body:
```
{
    "text": "Put input here"
}
```

###### Example curl command:
```sh
curl http://localhost:8888/convert/tts \
--header "Content-Type: application/json" \
-d '{ "text": "Put input here" }' \
--output "example.wav"
```

### Customize (Everything except for "text" is optional)
Response: .wav

###### Post body:
```
{
    "text": "input",
    "speed": "speed",
    "language": "language",
    "speaker_id": "speaker_id"
}
```

###### Example curl command:
```sh
curl http://localhost:8888/convert/tts \
--header "Content-Type: application/json" \
-d '{
  "text": "Put input here",
  "speed": "0.5",
  "language": "EN",
  "speaker_id": "EN-BR"
}' \
--output "example.wav"
```

## Acknowledgement

This is just an API server for the awesome work of [MeloTTS](https://github.com/myshell-ai/MeloTTS) from [MyShell](https://github.com/myshell-ai)
