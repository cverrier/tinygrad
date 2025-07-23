# from tinygrad.shape.view import View
# a = View.create(shape=(3, 2), strides=(2, 1))
#
# a = a.permute((1, 0))
# print("Shape:", a.shape)
# print("Strides:", a.strides)
#
# a = a.reshape((3, 2))
# print(a)

from tinygrad.shape.shapetracker import ShapeTracker

a = ShapeTracker.from_shape((3, 2))
a = a.permute((1, 0))
a = a.reshape((3, 2))
print(a)
idx, valid = a.to_indexed_uops()
print(idx.render())
