from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

# a = View.create(shape=(2, 2), strides=(2, 1))
# idx, valid = a.to_indexed_uops()
# print(idx.render())

a = ShapeTracker.from_shape((3, 2))
a = a.permute((1, 0))
a = a.reshape((3, 2))
print(a)
