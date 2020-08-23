# LinkedList

## 2. Add Two Numbers
```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode head = dummy;
        int carry = 0;
        while(l1 != null || l2 != null){
            int x = (l1 == null) ? 0 : l1.val;
            int y = (l2 == null) ? 0 : l2.val;
            int sum = carry + x + y;
            carry = sum / 10;
            head.next = new ListNode(sum % 10);
            head = head.next;
            l1 = (l1 == null) ? null : l1.next;
            l2 = (l2 == null) ? null : l2.next;
        }
        if (carry != 0) head.next = new ListNode(carry);
        return dummy.next;
    }
}
```
