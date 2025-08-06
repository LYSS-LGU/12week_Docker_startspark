# VM 하드디스크 Extend(디스크확장)

생성 일시: 2025년 8월 6일 오전 9:11
주*일차: 13주60일차

현재 VMware로 생성한 Ubuntu 가상머신의 **루트 파티션(`/`)** 공간이 **거의 가득 찬 상태(99%)**

```bash
devuser@ubuntu2204:~$ df -h
Filesystem                         Size  Used Avail Use% Mounted on
tmpfs                              388M  1.8M  386M   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv   19G   18G  194M  99% /
tmpfs                              1.9G     0  1.9G   0% /dev/shm
tmpfs                              5.0M     0  5.0M   0% /run/lock
/dev/sda2                          2.0G  252M  1.6G  14% /boot
tmpfs                              388M  4.0K  388M   1% /run/user/1000

```

## 디스크 확장하기

### 1. VMWare에서 디스크 확장

1. **VM을 완전히 종료한다.**
    
    ![image.png](image.png)
    
2.  VM 메뉴 - Settings - Harddisk 선택
3. 현재 화면에서 **"Expand..." 버튼 클릭**
    
    ![image.png](image%201.png)
    
4. 새로운 크기 입력 (예: 60GB 또는 80GB) → 80GB 입력 → Expand 선택
    
    ![image.png](image%202.png)
    
5. 확장 완료

![image.png](image%203.png)

1. VM 시작
    
    ![image.png](image%204.png)
    

### 2. Ubuntu에서 파티션 확장

VM 부팅 후 다음 명령들을 순서대로 실행:

<aside>
💡

```bash
# 1. 현재 상태 확인
df -h
lsblk

# 2. 파티션 테이블 새로고침
sudo partprobe
```

</aside>

```bash
devuser@ubuntu2204:~$ df -h
Filesystem                         Size  Used Avail Use% Mounted on
tmpfs                              388M  1.6M  387M   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv   19G   17G  1.1G  95% /
tmpfs                              1.9G     0  1.9G   0% /dev/shm
tmpfs                              5.0M     0  5.0M   0% /run/lock
/dev/sda2                          2.0G  252M  1.6G  14% /boot
tmpfs                              388M  4.0K  388M   1% /run/user/1000
devuser@ubuntu2204:~$ lsblk
NAME                      MAJ:MIN RM  SIZE RO TYPE MOUNTPOINTS
loop0                       7:0    0 63.9M  1 loop /snap/core20/2318
loop1                       7:1    0 63.8M  1 loop /snap/core20/2599
loop2                       7:2    0   87M  1 loop /snap/lxd/29351
loop3                       7:3    0 38.8M  1 loop /snap/snapd/21759
loop4                       7:4    0 89.4M  1 loop /snap/lxd/31333
loop5                       7:5    0 49.3M  1 loop /snap/snapd/24792
sda                         8:0    0   80G  0 disk 
├─sda1                      8:1    0    1M  0 part 
├─sda2                      8:2    0    2G  0 part /boot
└─sda3                      8:3    0   38G  0 part 
  └─ubuntu--vg-ubuntu--lv 253:0    0   19G  0 lvm  /
sr0                        11:0    1    2G  0 rom  
devuser@ubuntu2204:~$ sudo partprobe
[sudo] password for devuser: 
Warning: Unable to open /dev/sr0 read-write (Read-only file system).  /dev/sr0 has been opened read-only.
devuser@ubuntu2204:~$
```

<aside>
💡

```bash
# 3. 파티션 크기 조정 (parted 사용)
sudo parted /dev/sda
```

</aside>

```bash
devuser@ubuntu2204:~$ sudo parted /dev/sda
GNU Parted 3.4
Using /dev/sda
Welcome to GNU Parted! Type 'help' to view a list of commands.
(parted) 
```

parted 프롬프트에서:

<aside>
💡

```bash
(parted) print free
(parted) resizepart 3 100%
(parted) quit
```

</aside>

```bash
(parted) print free                                                       
Model: VMware, VMware Virtual S (scsi)
Disk /dev/sda: 85.9GB
Sector size (logical/physical): 512B/512B
Partition Table: gpt
Disk Flags: 

Number  Start   End     Size    File system  Name  Flags
        17.4kB  1049kB  1031kB  Free Space
 1      1049kB  2097kB  1049kB                     bios_grub
 2      2097kB  2150MB  2147MB  ext4
 3      2150MB  42.9GB  40.8GB
        42.9GB  85.9GB  43.0GB  Free Space

(parted) resizepart 3 100%                                                
(parted) quit                                                             
Information: You may need to update /etc/fstab.
```

<aside>
💡

```bash
# 4. 물리 볼륨 확장
sudo pvresize /dev/sda3

# 5. 논리 볼륨 확장
sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv

# 6. 파일시스템 확장
sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv
```

</aside>

```bash
devuser@ubuntu2204:~$ sudo pvresize /dev/sda3
  Physical volume "/dev/sda3" changed
  1 physical volume(s) resized or updated / 0 physical volume(s) not resized
devuser@ubuntu2204:~$ sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv
  Size of logical volume ubuntu-vg/ubuntu-lv changed from <19.00 GiB (4863 extents) to <78.00 GiB (19967 extents).
  Logical volume ubuntu-vg/ubuntu-lv successfully resized.
devuser@ubuntu2204:~$ sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv
  New size (19967 extents) matches existing size (19967 extents).
devuser@ubuntu2204:~$ sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv
resize2fs 1.46.5 (30-Dec-2021)
Filesystem at /dev/mapper/ubuntu--vg-ubuntu--lv is mounted on /; on-line resizing required
old_desc_blocks = 3, new_desc_blocks = 10
The filesystem on /dev/mapper/ubuntu--vg-ubuntu--lv is now 20446208 (4k) blocks long.
```

<aside>
💡

# 7. 결과 확인
df -h

</aside>

```bash
devuser@ubuntu2204:~$ df -h
Filesystem                         Size  Used Avail Use% Mounted on
tmpfs                              388M  1.6M  387M   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv   77G   17G   57G  23% /
tmpfs                              1.9G     0  1.9G   0% /dev/shm
tmpfs                              5.0M     0  5.0M   0% /run/lock
/dev/sda2                          2.0G  252M  1.6G  14% /boot
tmpfs                              388M  4.0K  388M   1% /run/user/1000
```

# 도커 컨테이너 다시 올리기

```bash
# 기존 컨테이너 완전 정리
docker compose down

# 다시 시작
docker compose up -d
```