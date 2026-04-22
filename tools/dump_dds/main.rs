fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dump_dds <in.dds> <out.png>");
        std::process::exit(1);
    }
    let bytes = std::fs::read(&args[1]).expect("read");
    let dds = image_dds::ddsfile::Dds::read(&bytes[..]).expect("parse dds");
    let img = image_dds::image_from_dds(&dds, 0).expect("decode");
    img.save(&args[2]).expect("save png");
    // Also print a few pixel values so we can see if they're grey
    let (w, h) = (img.width(), img.height());
    let samples = [(w/4, h/4), (w/2, h/2), (3*w/4, 3*h/4)];
    for (x, y) in samples {
        let p = img.get_pixel(x, y);
        println!("pixel ({}, {}): {:?}", x, y, p);
    }
}
