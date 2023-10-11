[485](#1)&nbsp;&nbsp;&nbsp; [27](#1)
<span id="1"></span>
## 485 最大连续个数
给定一个二进制数组 nums ， 计算其中最大连续 1 的个数。
```
示例 1：
输入：nums = [1,1,0,1,1,1]
输出：3
```
```java
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int maxCount = 0, count = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                count++;
            } else {
                maxCount = Math.max(maxCount, count);
                count = 0;
            }
        }
        maxCount = Math.max(maxCount, count);
        return maxCount;
    }
}
```
<span id="2"></span>
## 27 移除元素
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
