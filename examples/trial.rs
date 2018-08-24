trait Foo {
    fn hello(&self);
}

struct Bar {}

impl Foo for Bar {
    fn hello(&self) {
        println!("Hello!");
    }
}

fn foo1<F: Foo + ?Sized>(f: &F) {
    f.hello();
}

//fn foo2<F: Foo + ?Sized>(f: &F) {
//fn foo2(f: &Foo) {
//}

fn main() {
    let foo: &Foo = &Bar {}; // type-erasure
    foo1(foo);
}
