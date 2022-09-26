mkdir libs 
chdir libs 
git clone https://gitlab.com/quantum-ket/libket.git || git -C libket pull
git clone https://gitlab.com/quantum-ket/kbw.git || git -C kbw pull

chdir libket
cargo build --release
chdir ../kbw
cargo build --release
chdir ../..

mkdir ket\clib\libs
copy libs\libket\target\release\ket.dll ket\clib\libs
copy libs\kbw\target\release\kbw.dll ket\clib\libs