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
