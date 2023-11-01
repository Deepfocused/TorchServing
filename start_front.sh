docker run --gpus all \
  -it \
  --rm \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p8080:8080 \
  -p8081:8081 \
  -p8082:8082 \
  -p7070:7070 \
  -p7071:7071 \
  -e QT_X11_NO_MITSHM=1 \
  -e DISPLAY \
  -e "TERM=xterm-color" \
  -v ~/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --name torchserve \
  -v $(pwd):/home/hae/models \
  jonggon/torchserve_opencv:0.0.0 \
  torchserve --start  --ncs --ts-config config.properties --model-store . --models facedetector.mar --foreground
