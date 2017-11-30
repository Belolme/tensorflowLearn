class Animal:
    def __init__(self, clazz):
        self.clazz = clazz

    def eat(self):
        print("Animal eat")


class Person():
    """
    person doc
    """
    internalFile = "hello"  # class filed

    def __init__(self, name):
        self.internalFile = name  # object filed

    def sayHi(self):
        """
        if return segment has no value, the method return None directry.
        """
        # 在 Python 中，不过是在类内部还是外部，引用一个变量都需要通过类名或者 self 进行引用。
        # 同理，引用方法也一样
        print("Hello, how are you %s" % Person.internalFile)
        if False:
            return 'hi'
        else:
            return

    def sayHello(self):
        self.sayHi()

    def eat(self):
        print("pserson eat")


class Student(Animal, Person):
    """
    Student class, extend from Person
    Python 支持多重继承，当需要多重继承的时候，
    需要在其后加加上多个元素的 tuple 就可以了。

    在多重继承中，如果子类没有找到这一个方法，会依照
    顺序从继承的父类中依次查找。
    """

    def __init__(self, name):
        """
        python 没有强制调用父类的构造方法，父类的构造方法需要在子类中手动调用
        """
        Person.__init__(self, name)


p = Student('Amy')
print(p.eat())
