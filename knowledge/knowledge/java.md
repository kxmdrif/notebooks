# java

## 1. 单例模式

### 1.1 双重检查模式

```java
public class Singleton {  
    private volatile static Singleton singleton;  //1:volatile修饰
    private Singleton (){}  
    public static Singleton getSingleton() {  
    if (singleton == null) {  //2:减少不要同步，优化性能
        synchronized (Singleton.class) {  // 3：同步，线程安全
        if (singleton == null) {  
            singleton = new Singleton();  //4：创建singleton 对象
        }  
        }  
    }  
    return singleton;  
    }  
}

```
1. 延迟初始化。和懒汉模式一致，只有在初次调用静态方法getSingleton，才会初始化signleton实例。
2. 性能优化。同步会造成性能下降，在同步前通过判读singleton是否初始化，减少不必要的同步开销。
3. 线程安全。同步创建Singleton对象，同时注意到静态变量singleton使用volatile修饰。


为什么要使用volatile修饰？

虽然已经使用synchronized进行同步，但在第4步创建对象时，会有下面的伪代码：
```java
memory=allocate(); //1：分配内存空间
ctorInstance();   //2:初始化对象
singleton=memory; //3:设置singleton指向刚排序的内存空间
```
当线程A在执行上面伪代码时，2和3可能会发生重排序，因为重排序并不影响运行结果，还可以提升性能，所以JVM是允许的。如果此时伪代码发生重排序，步骤变为1->3->2,线程A执行到第3步时，线程B调用getsingleton方法，在判断singleton==null时不为null，则返回singleton。但此时singleton并还没初始化完毕，线程B访问的将是个还没初始化完毕的对象。**当声明对象的引用为volatile后，伪代码的2、3的重排序在多线程中将被禁止!**

### 1.2  静态内部类模式
```java
public class Singleton { 
    private Singleton(){
    }
      public static Singleton getSingleton(){  
        return Inner.instance;  
    }  
    private static class Inner {  
        private static final Singleton instance = new Singleton();  
    }  
} 

```
1. 实现代码简洁。
2. 延迟初始化。调用getSingleton才初始化Singleton对象。
3. 线程安全。JVM在执行类的初始化阶段，会获得一个可以同步多个线程对同一个类的初始化的锁。

如何实现线程安全？
线程A和线程B同时试图获得Singleton对象的初始化锁，假设线程A获取到了，那么线程B一直等待初始化锁。线程A执行类初始化，就算双重检查模式中伪代码发生了重排序，也不会影响线程A的初始化结果。初始化完后，释放锁。线程B获得初始化锁，发现Singleton对象已经初始化完毕，释放锁，不进行初始化，获得Singleton对象。

**注:  加载一个类时，其内部类(静态和非静态)不会同时被加载。一个类被加载，当且仅当其某个静态成员（静态域、构造器、静态方法等）被调用时发生。** 

### 1.3 单例模式

```java
public enum Singleton {  
    INSTANCE;  
    public void whateverMethod() {  
    }  
}
/*
使用： Singleton singleton = Singleton.INSTANCE;
枚举类的构造方法是私有的
*/
```
1. 反射安全(JVM 会阻止反射获取枚举类的私有构造方法)
2. 序列化/反序列化安全
3. 线程安全
4. 写法简单
5. 不可延迟实例化

## 2. 垃圾回收

年轻代中的GC

HotSpot JVM把年轻代分为了三部分：1个Eden区和2个Survivor区（分别叫from和to）。默认比例为8：1,为啥默认会是这个比例，接下来我们会聊到。一般情况下，新创建的对象都会被分配到Eden区(一些大对象特殊处理),这些对象经过第一次Minor GC后，如果仍然存活，将会被移到Survivor区。对象在Survivor区中每熬过一次Minor GC，年龄就会增加1岁，当它的年龄增加到一定程度时，就会被移动到年老代中。

因为年轻代中的对象基本都是朝生夕死的(80%以上)，所以在年轻代的垃圾回收算法使用的是复制算法，复制算法的基本思想就是将内存分为两块，每次只用其中一块，当这一块内存用完，就将还活着的对象复制到另外一块上面。复制算法不会产生内存碎片。

在GC开始的时候，对象只会存在于Eden区和名为“From”的Survivor区，Survivor区“To”是空的。紧接着进行GC，Eden区中所有存活的对象都会被复制到“To”，而在“From”区中，仍存活的对象会根据他们的年龄值来决定去向。年龄达到一定值(年龄阈值，可以通过-XX:MaxTenuringThreshold来设置)的对象会被移动到年老代中，没有达到阈值的对象会被复制到“To”区域。经过这次GC后，Eden区和From区已经被清空。这个时候，“From”和“To”会交换他们的角色，也就是新的“To”就是上次GC前的“From”，新的“From”就是上次GC前的“To”。不管怎样，都会保证名为To的Survivor区域是空的。Minor GC会一直重复这样的过程，直到“To”区被填满，“To”区被填满之后，会将所有对象移动到年老代中。