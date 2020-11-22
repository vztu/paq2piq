from paq2piq_standalone import *

model = InferenceModel(RoIPoolModel(), 'RoIPoolModel.pth')
model.blk_size = (3,5)
q = model.predict_from_file("Picture1.jpg")
print(q)