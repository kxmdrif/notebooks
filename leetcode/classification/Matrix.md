# Matrix

## 48. Rotate Image
**solution**

```java
class Solution {
    public void rotate(int[][] matrix) {
        /*
        A  B         C  A
                ->            
        C  D         D  B
          |
          |
          |          A  C         C  A
          |------>          ->  
                     B  D         D  B
        */
        int m = matrix.length;
        if (m == 0) return;
        int n = matrix[0].length;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if (i < j){
                    int tmp = matrix[i][j];
                    matrix[i][j] = matrix[j][i];
                    matrix[j][i] = tmp;
                }
            }
        }
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n / 2; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = tmp;
            }
        }
    }
}
```

## 79. Word Search

**solution**
DFS

```java
class Solution {
   
    public boolean exist(char[][] board, String word) {
        
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                boolean tmp = dfs(board, word, 0, i, j);
                if (tmp) return true;
            }
        }
        return false;
    }
    
    private boolean dfs(char[][] board, String word, int begin,
                         int i, int j){
        //begin == word.length() must be in front of 
        //the next 'if' statement, consider baord = [['a']]
        //and word = "a"
        if (begin == word.length()) return true;
        if (i < 0 || i >= board.length
           || j < 0 || j >= board[0].length
           || word.charAt(begin) != board[i][j]) return false;
        
        char tmp = board[i][j];
        //use board itself to mark used
        //we mark board[i][j] instead of the next position
        //or the result will be always false
        board[i][j] = ' ';
        boolean res = dfs(board, word, begin + 1, i - 1, j)
            || dfs(board, word, begin + 1, i + 1, j)
            || dfs(board, word, begin + 1, i, j - 1)
            || dfs(board, word, begin + 1, i, j + 1);
        board[i][j] = tmp;
        return res;
    }
 
}
```

## 240. Search a 2D Matrix II
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.
example: 
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
**solution**
- Time Complexity : O(m + n)
- Space Complexity: O(1)

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        if (m == 0) return false;
        int n = matrix[0].length;
        if (n == 0) return false;
        int r = 0, c = n - 1;
        while(r < m && c >= 0){
            if (matrix[r][c] == target) return true;
            if (matrix[r][c] > target) c--;
            else r++;
        }
        return false;
    }
}
```