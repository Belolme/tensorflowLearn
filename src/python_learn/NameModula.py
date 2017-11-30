
# 每个 Python 模块都有他的 __name__,如果他是 __main__,
# 说明这个模块被用户单独运行，我们可以进行相应恰当的操作。
if __name__ == '__main__':
    print("is running by itslef")
else:
    print("is imported from another module")

a = 1

# 这一个函数列出这一个模块内所有定义的 variables
print(dir())

del a
print(dir())
