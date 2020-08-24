# Design

## 146. LRU Cache
**solution**
Approach: HashMap + Double LinkedList
(regard map and linkedlist as independent structure)

```java
class LRUCache {
    class Node{
        int key;
        int value;
        Node next;
        Node pre;
        Node(int key, int value){
            this.key = key;
            this.value = value;
        }
        Node(){
            this(0, 0);
        }
    }
    private int capacity;
    private int size;
    //dummy node, not in map
    //head end is the newest
    private Node head;
    private Node tail;
    private Map<Integer, Node> map;
    public LRUCache(int capacity) {
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.pre = head;
        this.capacity = capacity;
        this.size = 0;
        map = new HashMap<>();
    }
    
    public int get(int key) {
        Node node = map.get(key);
        if (node == null)
            return -1;
        
        update(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null){
            node.value = value;
            update(node);
        }
        else{
            node = new Node(key, value);
            map.put(key, node);
            add(node);
            this.size++;
        }
        //First add, then decide to remove or not;
        if (this.size > capacity){
            Node toDel = tail.pre;
            //attention: tail.pre will be changed 
            //after remove,
            //so remove(tail.pre) then map.remove(tail.pre.key)
            //is wrong!
            remove(toDel);
            map.remove(toDel.key);
            this.size--;
        }
    }
    
    private void update(Node node){
        remove(node);
        add(node);
    }
    
    private void add(Node node){
        Node tmp = head.next;
        node.pre = head;
        node.next = tmp;
        head.next = node;
        tmp.pre = node;
    }
    private void remove(Node node){
        node.pre.next = node.next;
        node.next.pre = node.pre;
        node.pre = null;
        node.next = null;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```