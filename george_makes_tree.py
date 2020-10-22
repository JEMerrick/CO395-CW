leftnode = {
      "def": "I am left node"
}
print("left:" + str(leftnode))
rightnode = {
       "def": "I am right node"
}
print("right:" + str(rightnode))
bigboynode = {
      "left": leftnode,
      "right": rightnode
}
print("bigboy:" + str(bigboynode))
print("left:" + str(bigboynode["left"]))
print("right:" + str(bigboynode["right"]))
print("right def:" + str(bigboynode["right"]["def"]))
