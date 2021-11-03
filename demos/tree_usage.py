"""
"""
from cluster.hierarchical_clustering import Tree
from cluster.hierarchical_clustering import Node

root_node = Node('root_node', current_level=0)
node1 = Node('node1')
node2 = Node('node2')
node3 = Node('node3')
node4 = Node('node4')
node5 = Node('node5')
node6 = Node('node6')
node7 = Node('node7')
node8 = Node('node8')
node9 = Node('node9')

root_node.close()
node1.close()
node2.close()
node3.close()
node4.close()
node5.close()
node6.close()
node7.open()
node8.open()
node9.open()


tree = Tree(root_node)
root_node.addChildNode(node1)
root_node.addChildNode(node2)
root_node.addChildNode(node3)
node3.addChildNode(node4)
node3.addChildNode(node5)
node3.addChildNode(node6)
node6.addChildNode(node7)
node6.addChildNode(node8)
node6.addChildNode(node9)


print('PRINTING ALL NODES: ')
for node in tree.getAllNodes():
    print(node)

print('PRINTING OPEN NODES: ')
for node in tree.getOpenNodes():
    print(node)
