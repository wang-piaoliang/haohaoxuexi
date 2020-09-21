# Docker与K8S了解一下？

10分钟看懂Docker和K8S \(王脸小: 写得爆炸好\)

[https://zhuanlan.zhihu.com/p/53260098](https://zhuanlan.zhihu.com/p/53260098)

K8s理解初识到应用到理解 （写得太好惹）

[https://juejin.im/post/5c98b1785188252d665f57be\#heading-0](https://juejin.im/post/5c98b1785188252d665f57be#heading-0)

## **Docker** <a id="Docker"></a>

轻量级的虚拟化技术

1. Build, Ship and Run
2. Build once，Run anywhere

### 三大核心概念 <a id="%E4%B8%89%E5%A4%A7%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5"></a>

1. 镜像（Image）
2. 容器（Container）
3. 仓库（Repository）

那个放在包里的“镜像”，就是Docker镜像。而我的背包，就是Docker仓库。我在空地上，用魔法造好的房子，就是一个Docker容器。

### 容器和虚拟机的对比 <a id="%E5%AE%B9%E5%99%A8%E5%92%8C%E8%99%9A%E6%8B%9F%E6%9C%BA%E7%9A%84%E5%AF%B9%E6%AF%94"></a>

## **Kubernetes** <a id="Kubernetes"></a>

K8S，就是基于容器的集群管理平台。

一个K8S系统，通常称为一个K8S集群（Cluster）

这个集群主要包括两个部分：

1. 一个Master节点（主节点）
2. 一群Node节点（计算节点）

### Master节点 <a id="Master%E8%8A%82%E7%82%B9"></a>

包括API Server、Scheduler、Controller manager、etcd。

* API Server是整个系统的对外接口，供客户端和其它组件调用，相当于“营业厅”。
* Scheduler负责对集群内部的资源进行调度，相当于“调度室”。
* Controller manager负责管理控制器，相当于“大总管”。

### Node节点 <a id="Node%E8%8A%82%E7%82%B9"></a>

包括Docker、kubelet、kube-proxy、Fluentd、kube-dns（可选），还有就是**Pod**。

Pod是Kubernetes最基本的操作单元。一个Pod代表着集群中运行的一个进程，它内部封装了一个或多个紧密相关的容器。除了Pod之外，K8S还有一个**Service**的概念，一个Service可以看作一组提供相同服务的Pod的对外访问接口。这段不太好理解，跳过吧。

### 动手创建集群 <a id="%E5%8A%A8%E6%89%8B%E5%88%9B%E5%BB%BA%E9%9B%86%E7%BE%A4"></a>

在CentOS上部署kubernetes集群

[https://jimmysong.io/kubernetes-handbook/practice/install-kubernetes-on-centos.html](https://jimmysong.io/kubernetes-handbook/practice/install-kubernetes-on-centos.html)

和我一步步部署 kubernetes 集群

[https://k8s-install.opsnull.com/](https://k8s-install.opsnull.com/)

Prepare an application for Azure Kubernetes Service \(AKS\)

[https://docs.microsoft.com/en-us/azure/aks/tutorial-kubernetes-prepare-app](https://docs.microsoft.com/en-us/azure/aks/tutorial-kubernetes-prepare-app)

Quickstart: Deploy an Azure Kubernetes Service \(AKS\) cluster using the Azure portal

[https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough-portal](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough-portal)

