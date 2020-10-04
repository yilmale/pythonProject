def M1(self):
    print("Hello-M1")
    self.i = 15

def M2(self):
    print("Hello-M2")
    self.i = 20

ParentClass = type('ParentClass',(),{})
MyClass = type('MyClass',(ParentClass,),{
    'M1': M1,
    'i': 10
})

x = MyClass()
print(x.i)
x.M1()
print(x.i)

d = MyClass.__dict__
print(d)
e=d.copy()
e.update({'M2': M2})
print(e)

MyClass= type('MyClass',(ParentClass,),e)

y=MyClass()
y.M2()
print(y.i)