inputs=[0,0,0,0]

weights=hAs

for i in weights:
    print(i)


print("\n")
out=activate(0, weights, inputs)

print (out)

out2 =transfer(out)

print(out2)

out3=forward_propagate(weights, inputs)


print(out3)



print(" \n Steps : ")
for i in steps :
    print(i)