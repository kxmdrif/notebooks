# Array

## 4. Median of Two Sorted Arrays

**solution**
O(log n)

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        //for an array with length n, whether n is odd or even, 
        //the Median is always the average of the (n + 1)/2 th and the (n + 2)/2 th
        int left = (m + n + 1) / 2, right = (m + n + 2) / 2;
        return (findKth(nums1, nums2, 0, 0, left) + 
            findKth(nums1, nums2, 0, 0, right)) / 2.0;
    }
    
    // find kth smallest item in two mergered array
    public int findKth(int[] nums1, int[] nums2, int i, int j, int k){
        if (i >= nums1.length) return nums2[j + k - 1];
        if (j >= nums2.length) return nums1[i + k - 1];
        if (k == 1) return Math.min(nums1[i], nums2[j]);
        
        //if midVal1 is out of range, kth might be in nums1[i, )
        //but kth couldn't be in nums2[i, i + k / 2) 
        //so we use INT MAX to prevent nums1[i, ) from being elimilated
        int midVal1 = (i + k / 2 - 1) < nums1.length
            ? nums1[i + k / 2 - 1] : Integer.MAX_VALUE;
        int midVal2 = (j + k / 2 - 1) < nums2.length
            ? nums2[j + k / 2 - 1] : Integer.MAX_VALUE;
        
        return midVal1 < midVal2 
            ? findKth(nums1, nums2, i + k / 2, j, k - k / 2)
            : findKth(nums1, nums2, i, j + k / 2, k - k / 2);
        //k - k / 2 couldn't be replaced by k / 2!!!
    }
    
    
}
```