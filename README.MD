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

Upsampling é um técnica de processamento digital de sinais para aumentar artificialmente a taxa de amostragem em N vezes, inserindo um número N-1 de zeros entre as amostras originais do sinal, e passando o conjunto obtido por um filtro de reconstrução, que nada mais é que um filtro do tipo passa-baixas

Podemos trabalhar esse parametro para dar foco em faces, redimencionando para ter melhor precisão na deteccão Ex : 

```
deteccoes = detector_face_hog(imagem, 2)
```

## MMOD - Max Margin Object Detection
Tecnica de tetecção de objetos através da margem
Da area de Deep Learning precisa de GPU pois demanda processamento alto de maquina

Este detecta faces em todos os angulos, que não estão em posições frontal

Pois usa algoritimos redes neurais convolucionais

### Uma Rede Neural Convolucional (ConvNet / Convolutional Neural Network / CNN) 

É um algoritmo de Aprendizado Profundo que pode captar uma imagem de entrada, atribuir importância (pesos e vieses que podem ser aprendidos) a vários aspectos / objetos da imagem e ser capaz de diferenciar um do outro.


Esse Detector retorna alem das detecções de forma mais precisa em em varios angulos, mas tambem o nivel de confiança na variavel `face.confidence`., conforme exemplo abaixo : 

```
imagem = cv2.imread('/content/people1.jpg')
imagem = cv2.resize(imagem, (600, 400))
detector_cnn_mmod = dlib.cnn_face_detection_model_v1('/content/mmod_human_face_detector.dat')

deteccoes = detector_cnn_mmod(imagem, 1)
for face in deteccoes:
  #print(face)
  l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
  print(c)
  cv2.rectangle(imagem, (l, t), (r, b), (255,255,0), 2)
cv2_imshow(imagem)
```

## OpenCV - SSD: Single Shot Multibox Detector

É uma evolução dos Algoritmos anteriores e possui mais precisão e até desempenho consideranto tarefas complexas de detecção.

Usa rede neural artificial, tambem correlacionada a Multibox  fazendo regressão para gerar bounding boxes automaticos

Podemos utilizar modelos treinados pelos TensowFlor pe Pytork ou pelo Caffe conforme exemplo abaixo : 

```
arquivo_modelo = '/content/res10_300x300_ssd_iter_140000.caffemodel'
arquivo_prototxt = '/content/deploy.prototxt.txt'
network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)
conf_min = 0.5
imagem = cv2.imread('/content/people2.jpg')
imagem.shape
(h, w) = imagem.shape[:2]
print(h, w)
cv2_imshow(imagem)
# Documentation: https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300,300)), 1.0, (300,300), (104.0, 117.0, 123.0))

network.setInput(blob) # Envio a imagem para a rede neural
deteccoes = network.forward() # Faz a rede neural percorrer as camadas para retornar as detecções

for i in range(0, deteccoes.shape[2]):
  confianca = deteccoes[0, 0, i, 2]
  if confianca > conf_min:
    #print(confianca)
    text_conf = "{:.2f}%".format(confianca * 100)
    box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
    #print(box)
    (start_x, start_y, end_x, end_y) = box.astype(int)
    #print(start_x, start_y, end_x, end_y)
    cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
    cv2.putText(imagem, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2_imshow(imagem)
```

É um pouco mais complexo mas é muito mais preciso

Aumentando o tamanho da imagem a confiança fica melhor : 

Ex : 

```
path_imagem = '/content/people3.jpg'
detecta_face_ssd(network, path_imagem, 500)


def detecta_face_ssd(net, path_imagem, tamanho = 300):
  imagem = cv2.imread(path_imagem)
  (h, w) = imagem.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (tamanho, tamanho)), 2.0, (tamanho, tamanho), (104.0, 117.0, 123.0))
  net.setInput(blob)
  deteccoes = net.forward()

  for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]

    if confianca > conf_min:
      box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
      (start_x, start_y, end_x, end_y) = box.astype("int")

      text_conf = "{:.2f}%".format(confianca * 100)
      cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
      cv2.putText(imagem, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

  cv2_imshow(imagem)
```

## Detecção em Videos

Podemos utilizar a lib cv2

Ex : 

```
arquivo_video = '/content/gabydancandocommusica.mp4'
cap = cv2.VideoCapture(arquivo_video)

conectado, video = cap.read()
print(conectado)

```

Podemos regravar o video com os bounded Box : 

```
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24 ## podemos reconfigurar para o video ficar mais rapido ou mais lento
video_saida = cv2.VideoWriter(arquivo_resultado, fourcc, fps, (video_largura, video_altura))
```

Comando util para converter o video de avi para mp4

```
!ffmpeg -y -loglevel panic -i resultado.avi resultado.mp4
```


Com deep learning é muito melhor os resultados de deteccçoes do que o haarcascade Exemplo de implementação

```
network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

def detecta_face_ssd(net, imagem, show_conf=True, tamanho = 300, conf_min = 0.7):
  (h, w) = imagem.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (tamanho, tamanho)), 1.0, (tamanho, tamanho), (104.0, 117.0, 123.0))
  net.setInput(blob)
  deteccoes = net.forward()

  face = None
  for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0, 0, i, 2]
    if confianca > conf_min:
      box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
      (start_x, start_y, end_x, end_y) = box.astype("int")
  
      cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
      if show_conf:
        text_conf = "{:.2f}%".format(confianca * 100)
        cv2.putText(imagem, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

  return imagem

while cv2.waitKey(1) < 0:
  conectado, frame = cap.read()
  if not conectado:
    break
  if max_frames > -1 and frame_atual > max_frames:
    break
  (H, W) = frame.shape[:2]
  t = time.time()
  if largura_maxima is not None:
    frame = cv2.resize(frame, (video_largura, video_altura))
  frame_processado = detecta_face_ssd(network, frame, True, 500)
  cv2.putText(frame, " frame processado em {:.2f} segundos".format(time.time() - t), (20, video_altura - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250,250,250), 0, lineType = cv2.LINE_AA)
  video_saida.write(frame_processado)
  if frame_atual <= frames_show:
    cv2_imshow(cv2.resize(frame_processado, (0,0), fx = 0.75, fy = 0.75))
  frame_atual += 1

print('Terminou!')
video_saida.release()
cv2.destroyAllWindows()

!ffmpeg -y -loglevel panic -i resultado.avi resultado.mp4

show_video('resultado.mp4')

```

Com isso é possivel ver o seu video com as respectivas deteccções
