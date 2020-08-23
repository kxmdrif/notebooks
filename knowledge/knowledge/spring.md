# spring

## 1. IoC初始化

```java
public void refresh() throws BeansException, IllegalStateException {
   // 来个锁，不然 refresh() 还没结束，你又来个启动或销毁容器的操作，那不就乱套了嘛
   synchronized (this.startupShutdownMonitor) {

      // 准备工作，记录下容器的启动时间、标记“已启动”状态、处理配置文件中的占位符
      prepareRefresh();

      // 这步比较关键，这步完成后，配置文件就会解析成一个个 Bean 定义，注册到 BeanFactory 中，
      // 当然，这里说的 Bean 还没有初始化，只是配置信息都提取出来了，
      // 注册也只是将这些信息都保存到了注册中心(说到底核心是一个 beanName-> beanDefinition 的 map)
      ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();

      // 设置 BeanFactory 的类加载器，添加几个 BeanPostProcessor，手动注册几个特殊的 bean
      // 这块待会会展开说
      prepareBeanFactory(beanFactory);

      try {
         // 【这里需要知道 BeanFactoryPostProcessor 这个知识点，Bean 如果实现了此接口，
         // 那么在容器初始化以后，Spring 会负责调用里面的 postProcessBeanFactory 方法。】

         // 这里是提供给子类的扩展点，到这里的时候，所有的 Bean 都加载、注册完成了，但是都还没有初始化
         // 具体的子类可以在这步的时候添加一些特殊的 BeanFactoryPostProcessor 的实现类或做点什么事
         postProcessBeanFactory(beanFactory);
         // 调用 BeanFactoryPostProcessor 各个实现类的 postProcessBeanFactory(factory) 方法
         invokeBeanFactoryPostProcessors(beanFactory);

         // 注册 BeanPostProcessor 的实现类，注意看和 BeanFactoryPostProcessor 的区别
         // 此接口两个方法: postProcessBeforeInitialization 和 postProcessAfterInitialization
         // 两个方法分别在 Bean 初始化之前和初始化之后得到执行。注意，到这里 Bean 还没初始化
         registerBeanPostProcessors(beanFactory);

         // 初始化当前 ApplicationContext 的 MessageSource，国际化这里就不展开说了，不然没完没了了
         initMessageSource();

         // 初始化当前 ApplicationContext 的事件广播器，这里也不展开了
         initApplicationEventMulticaster();

         // 从方法名就可以知道，典型的模板方法(钩子方法)，
         // 具体的子类可以在这里初始化一些特殊的 Bean（在初始化 singleton beans 之前）
         onRefresh();

         // 注册事件监听器，监听器需要实现 ApplicationListener 接口。这也不是我们的重点，过
         registerListeners();

         // 重点，重点，重点
         // 初始化所有的 singleton beans
         //（lazy-init 的除外）
         finishBeanFactoryInitialization(beanFactory);

         // 最后，广播事件，ApplicationContext 初始化完成
         finishRefresh();
      }

      catch (BeansException ex) {
         if (logger.isWarnEnabled()) {
            logger.warn("Exception encountered during context initialization - " +
                  "cancelling refresh attempt: " + ex);
         }

         // Destroy already created singletons to avoid dangling resources.
         // 销毁已经初始化的 singleton 的 Beans，以免有些 bean 会一直占用资源
         destroyBeans();

         // Reset 'active' flag.
         cancelRefresh(ex);

         // 把异常往外抛
         throw ex;
      }

      finally {
         // Reset common introspection caches in Spring's core, since we
         // might not ever need metadata for singleton beans anymore...
         resetCommonCaches();
      }
   }
}
```
1. 初始化 ApplicationContext
环境属性的初始化和验证，启动时间记录和相关标记设置，应用事件和监听者的初始化。
2. 准备好容器中的 BeanDefinition (eager-initializing beans)
对 BeanDefinition 的解析、扫描和注册，BeanDefinition 的扫描和注册大致可以分为 XML 和注解两种，两种方式各自使用的组件有所不同，该步骤的时间也可以在最前面。
3. 初始化 BeanFactory
准备好 BeanFactory 以供 ApplicationContext 进行使用，对接下来将要使用到的 Bean 进行实例化，资源进行准备，属性进行设置。
4. 注册 BeanPostProcessors BeanPostProcessors 是进行扩展的关键组件，需要在该步骤中进行注册，可分为两种类型: 一种是框架使用者提供的，用于特定业务功能的，另一种是框架开发者提供的，用于扩展框架功能。
5. 调用 BeanDefinitionRegistryPostProcessor, BeanDefinitionRegistryPostProcessor 是一种功能增强，可以在这个步骤添加新的 BeanDefinition 到 BeanFactory 中。
6. 调用 BeanFactoryPostProcessor, BeanFactoryPostProcessor 是一种功能增强，可以在这个步骤对已经完成初始化的 BeanFactory 进行属性覆盖，或是修改已经注册到 BeanFactory 的 BeanDefinition。
7. 初始化 MessageSource 和 ApplicationEventMulticaster, MessageSource 用于处理国际化资源，ApplicationEventMulticaster 是应用事件广播器，用于分发应用事件给监听者。
8. 初始化其他 Bean 和进行其他的的上下文初始化,主要用于扩展
9. 注册 ApplicationListener, 将 ApplicationListener 注册到 BeanFactory 中，以便后续的事件分发
10. 实例化剩余的 Bean 单例(非懒加载情况)
步骤 4 到 9 都对一些特殊的 Bean 进行了实例化，这里需要对所有剩余的单例 Bean 进行实例化
11. 启动完成, 资源回收，分发"刷新完成"事件。


## 2. SpringBoot启动过程

SpringBoot启动的时候，不论调用什么方法，都会构造一个SpringApplication的实例，然后调用这个实例的run方法，这样就表示启动SpringBoot。

在run方法调用之前，也就是构造SpringApplication的时候会进行初始化的工作，初始化的时候会做以下几件事：

1. 把参数sources设置到SpringApplication属性中，这个sources可以是任何类型的参数。本文的例子中这个sources就是MyApplication的class对象
2. 判断是否是web程序，并设置到webEnvironment这个boolean属性中
3. 找出所有的初始化器，默认有5个，设置到initializers属性中
4. 找出所有的应用程序监听器，默认有9个，设置到listeners属性中
5. 找出运行的主类(main class)


SpringApplication构造完成之后调用run方法，启动SpringApplication，run方法执行的时候会做以下几件事：

1. 构造一个StopWatch，观察SpringApplication的执行
2. 找出所有的SpringApplicationRunListener并封装到SpringApplicationRunListeners中，用于监听run方法的执行。监听的过程中会封装成事件并广播出去让初始化过程中产生的应用程序监听器进行监听
3. 构造Spring容器(ApplicationContext)，并返回
    3.1 创建Spring容器的判断是否是web环境，是的话构造AnnotationConfigEmbeddedWebApplicationContext，否则构造AnnotationConfigApplicationContext
    3.2 初始化过程中产生的初始化器在这个时候开始工作
    3.3 Spring容器的刷新(完成bean的解析、各种processor接口的执行、条件注解的解析等等)
4. 从Spring容器中找出ApplicationRunner和CommandLineRunner接口的实现类并排序后依次执行


## 3. Spring Bean的生命周期

Bean 容器找到配置文件中 Spring Bean 的定义。
Bean 容器利用 Java Reflection API 创建一个Bean的实例。
如果涉及到一些属性值 利用 set()方法设置一些属性值。
如果 Bean 实现了 BeanNameAware 接口，调用 setBeanName()方法，传入Bean的名字。
如果 Bean 实现了 BeanClassLoaderAware 接口，调用 setBeanClassLoader()方法，传入 ClassLoader对象的实例。
与上面的类似，如果实现了其他 *.Aware接口，就调用相应的方法。
如果有和加载这个 Bean 的 Spring 容器相关的 BeanPostProcessor 对象，执行postProcessBeforeInitialization() 方法
如果Bean实现了InitializingBean接口，执行afterPropertiesSet()方法。
如果 Bean 在配置文件中的定义包含 init-method 属性，执行指定的方法。
如果有和加载这个 Bean的 Spring 容器相关的 BeanPostProcessor 对象，执行postProcessAfterInitialization() 方法
当要销毁 Bean 的时候，如果 Bean 实现了 DisposableBean 接口，执行 destroy() 方法。
当要销毁 Bean 的时候，如果 Bean 在配置文件中的定义包含 destroy-method 属性，执行指定的方法。

## 4. Spring Bean作用域

五种作用域中，request、session 和 global session 三种作用域仅在基于web的应用中使用（不必关心你所采用的是什么web应用框架），只能用在基于 web 的 Spring ApplicationContext 环境。

1. singleton——唯一 bean 实例
当一个 bean 的作用域为 singleton，那么Spring IoC容器中只会存在一个共享的 bean 实例，并且所有对 bean 的请求，只要 id 与该 bean 定义相匹配，则只会返回bean的同一实例。 singleton 是单例类型(对应于单例模式)，就是在创建起容器时就同时自动创建了一个bean的对象，不管你是否使用，但我们可以指定Bean节点的 lazy-init=”true” 来延迟初始化bean，这时候，只有在第一次获取bean时才会初始化bean，即第一次请求该bean时才初始化。 每次获取到的对象都是同一个对象。注意，singleton 作用域是Spring中的缺省作用域。

2. prototype——每次请求都会创建一个新的 bean 实例
当一个bean的作用域为 prototype，表示一个 bean 定义对应多个对象实例。 prototype 作用域的 bean 会导致在每次对该 bean 请求（将其注入到另一个 bean 中，或者以程序的方式调用容器的 getBean() 方法）时都会创建一个新的 bean 实例。prototype 是原型类型，它在我们创建容器的时候并没有实例化，而是当我们获取bean的时候才会去创建一个对象，而且我们每次获取到的对象都不是同一个对象。根据经验，对有状态的 bean 应该使用 prototype 作用域，而对无状态的 bean 则应该使用 singleton 作用域。

3. request——每一次HTTP请求都会产生一个新的bean，该bean仅在当前HTTP request内有效
request只适用于Web程序，每一次 HTTP 请求都会产生一个新的bean，同时该bean仅在当前HTTP request内有效，当请求结束后，该对象的生命周期即告结束。

4. session——每一次HTTP请求都会产生一个新的 bean，该bean仅在当前 HTTP session 内有效
session只适用于Web程序，session 作用域表示该针对每一次 HTTP 请求都会产生一个新的 bean，同时该 bean 仅在当前 HTTP session 内有效.与request作用域一样，可以根据需要放心的更改所创建实例的内部状态，而别的 HTTP session 中根据 userPreferences 创建的实例，将不会看到这些特定于某个 HTTP session 的状态变化。当HTTP session最终被废弃的时候，在该HTTP session作用域内的bean也会被废弃掉。

5. globalSession
global session 作用域类似于标准的 HTTP session 作用域，不过仅仅在基于 portlet 的 web 应用中才有意义。Portlet 规范定义了全局 Session 的概念，它被所有构成某个 portlet web 应用的各种不同的 portle t所共享。在global session 作用域中定义的 bean 被限定于全局portlet Session的生命周期范围内。