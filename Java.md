## 数组

```java
int[] a=new int[]{1,2,3};
```
```java
int[] a=new int[3];
for(int i=0;i<a.length;i++){
  a[i]=i+1;}
System.out.println(Arrays.toString(a));
```
```java
ArrayList<integer> arr=new ArrList<>{}
for(int i=0;i<3;i++){
arr.add(i)}
System.out.println(Arrays.toString(a));
arr.add(2,99);//位置2插入元素99
int arr1=arr.get(1);//获取元素
arr.set(1,11);//更新
arr.remove(3);//删除
int arrsize=arr.size();//长度
for(int i=0;i<arr.size;i++){
int current=arr.get(i);
System.out.println(current);}//遍历
Arrays.sort(arr);//排序
```
### 冒泡排序
```java
import java.util.Arrays;  
  
public class Test {  
  public static void main(String[] args){  
     int[] arr1={3,6,2,8,11,4,8};  
     int[] arrSort=bubbleSort(arr1);  
     System.out.println(Arrays.toString(arrSort));  
  }  
  
  public static int[] bubbleSort(int[] arr) {    
    int n = arr.length;    
    for (int i = 0; i < n - 1; i++) {    
        for (int j = 0; j < n - i - 1; j++) {    
            if (arr[j] > arr[j + 1]) {    
                // 交换arr[j]和arr[j+1]    
                int temp = arr[j];    
                arr[j] = arr[j + 1];    
                arr[j + 1] = temp;    
            }    
        }   
        System.out.println(Arrays.toString(arr)); // 打印每一轮排序后的数组  
    }  
    return arr;    
}  
}
```
## 链表
```java
LinkedList<Integer> list=new LinkedList<>();
add.list(1);
add.list(2);
add.list(2,99);
System.out.println(list.toString())
int element=list.get(2);
int index=list.indexOf(99);
list.set(2,88);
list.remove(2);
int length=list.size();
```
