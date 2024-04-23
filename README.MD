# Detecção Facial

Descrição : Primeira parte antes de realizar um reconhecimento facial, os algoritmos costuma aplicar rotinas para detecção antes de qualquer associação de reconhecimento.

Aplicações : 

- Contagem de pessoas
- Auto foco em cameras
- Alarmes
- Rastreamento de objetos

Bounded Box - Area de detecção de face

# Reconhecimento

Aplicações : 

- Controle de acesso em areas restritas
- Desbloqueio de Celulares 
- Chamada Eletronica (Virtual ou presencial)

# Principais Libs para deteccção e reconhecimento de imagens : 

[OpenCV](./OpenCV.md) e [Dlib](./Dlib.md)

# Imagens

Curiosidades sobre imagens, toda imagem nada mais é do que uma cadeia de pixels

Cada pixel é representado pela escala RGB de 0 a 255. 

Imagens em escala cinza possuem sempre o mesmo valor no R G e B independente, sendo assim ocupam menos espaço e ajudam aos algoritimos de detecção e reconhecimento serem mais performaticos.

Existe um xml do haarcascade que por default consegue detectar faces.

[haarcascade_frontalface_default.xml](./haarcascade_frontalface_default.xml)

basta instanciar ele assim : 

```
detector_facial = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
```

E depois obter as detecções conforme abaixo : 

```
deteccoes = detector_facial.detectMultiScale(imagem_cinza)
```

