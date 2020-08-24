# Random

## 138. Copy List with Random Pointer
**solution**

Approach : 
insert copy node into the gap of two adjacent node,
then set the random pointer of the copy node, 
then get the result and recover the original list.

- Time complexity: O(n)
- Space complexity: O(1)

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
import java.util.*;
class Solution {
    public Node copyRandomList(Node head) {
        Node dummy = new Node(0);
        Node headptr = head;
        while(headptr != null){
            Node next = headptr.next;
            headptr.next = new Node(headptr.val);
            headptr.next.next = next;
            headptr = next;
        }
        headptr = head;
        while(headptr != null){
            Node random = headptr.random;
            headptr.next.random = random == null ? null : random.next;
            headptr = headptr.next.next;
        }
        headptr = head;
        Node resptr = dummy;
        while(headptr != null){
            
            resptr.next = headptr.next;
            headptr.next = headptr.next.next;
            headptr = headptr.next;
            resptr = resptr.next;
        }
        return dummy.next;
        
    }
}
```