# Tire

## 208. Implement Trie (Prefix Tree)
**solution**

```java
class Trie {

    class TrieNode{
        public boolean isWord;
        public TrieNode[] children;
        public TrieNode(){
            children = new TrieNode[26];
        }
    }
    /** Initialize your data structure here. */
    private TrieNode root;
    public Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode cur = root;
        for(char ch : word.toCharArray()){
            if (cur.children[ch - 'a'] == null)
                cur.children[ch - 'a'] = new TrieNode();
            cur = cur.children[ch - 'a'];
        }
        cur.isWord = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode cur = root;
        for(char ch : word.toCharArray()){
            if (cur.children[ch - 'a'] == null)
                return false;
            cur = cur.children[ch - 'a'];
        }
        return cur.isWord;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode cur = root;
        for(char ch : prefix.toCharArray()){
            if (cur.children[ch - 'a'] == null)
                return false;
            cur = cur.children[ch - 'a'];
        }
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

## 212. Word Search II
**solution**
Approach: Trie (a brute force way is invoking word Search I for every word)

```java
class Solution {
    
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        TrieNode root = buildTrie(words);
        for(int i = 0; i < board.length; i++)
            for(int j = 0; j < board[0].length; j++)
                dfs(board, i, j, root, res);
        return res;
    }
    //we can regard trie as a special string(compare to word search I)
    private void dfs(char[][] board, int i, int j,
                     TrieNode root, List<String> res){
        if (root.word != null){
            res.add(root.word);
            //avoid duplicate
            root.word = null;
        }
        if (i < 0 || i >= board.length
            || j < 0 || j >= board[0].length 
            || board[i][j] == ' ' 
            || root.next[board[i][j] - 'a'] == null)
            return;
        char ch = board[i][j];
        board[i][j] = ' ';
        dfs(board, i - 1, j, root.next[ch - 'a'], res);
        dfs(board, i + 1, j, root.next[ch - 'a'], res);
        dfs(board, i, j - 1, root.next[ch - 'a'], res);
        dfs(board, i, j + 1, root.next[ch - 'a'], res);
        board[i][j] = ch;
        
            
    }
    
    class TrieNode{
        TrieNode[] next;
        String word;
        public TrieNode(){
            next = new TrieNode[26];
        }
    }
    private TrieNode buildTrie(String[] words){
        TrieNode root = new TrieNode();
        for(String word : words){
            TrieNode cur = root;
            for(char ch : word.toCharArray()){
                if (cur.next[ch - 'a'] == null)
                    cur.next[ch - 'a'] = new TrieNode();
                cur = cur.next[ch - 'a'];
            }
            cur.word = word;
        }
        return root;
    }
        
        
}
```