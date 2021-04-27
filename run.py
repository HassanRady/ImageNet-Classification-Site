from model.model_factory import ModelFactory

model_factory = ModelFactory()

model_resnet = model_factory.get_resnet34()

print(model_resnet)
# print(model_resnet.predict("static/car1.jpg"))