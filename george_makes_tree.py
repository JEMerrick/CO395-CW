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
