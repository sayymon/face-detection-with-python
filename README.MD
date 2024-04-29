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

Redimencionar a imagem ajuda na detecção, se tiver com escala baixxa pode não identificar corretamente , mas se estiver com proporções muito grandes pode detectar faces alem do esperado como em camisetas fundo entre outros.

Para ajustar a escala de forma util podemos usar a função resize Ex : 

```
imagem = cv2.imread('/content/family_kgns.jpeg')
imagem = cv2.resize(imagem, (0,0), fx = 0.6, fy = 0.6)
imagem.shape
```

A função retangle ajuda a marcar a imagem com as faces detectadas ex : 

```
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
deteccoes = detector_facial.detectMultiScale(imagem_cinza)
for (x, y, w, h) in deteccoes:
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (100,255,255), 3)
cv2_imshow(imagem)
```

# Hog - Histogram of Oriented Gradient

É um algoritmo que trabalha com os histpgramas para conseguir identificar objetos e partes do corpo por exemplo usando indicadores , possui uma precisão melhor do que o Haarcascade.

usa a Dlib e não precisa converter em escala de cinza pq inclusive utiliza as cores

Podemos usar o : 
```
detector_face_hog = dlib.get_frontal_face_detector()
```
Para detectar rostos conforme abaixo : 

```
imagem = cv2.imread('/content/people2.jpg')
deteccoes = detector_face_hog(imagem)
for face in deteccoes:
  l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
  cv2.rectangle(imagem, (l, t), (r, b), (0,255,255), 2)
cv2_imshow(imagem)
```
